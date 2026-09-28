"""Test hàm thuần của core/speakers.py — không GPU, không đụng output/.

    cd vendor/VideoLingo && .venv/bin/python -m unittest discover -s tests -p 'test_speakers*.py'

Load module bằng đường dẫn file (không qua core/__init__.py vốn import torch/demucs).
"""
import importlib.util
import itertools
import os
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("vl_speakers", os.path.join(_HERE, "..", "core", "speakers.py"))
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


def W(text, start, end):
    return {"text": text, "start": start, "end": end}


def T(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def chars(text, start, step=0.2):
    """Mỗi ký tự một word liền nhau (như WhisperX tiếng Trung)."""
    out, t = [], start
    for ch in text:
        out.append(W(ch, round(t, 3), round(t + step, 3)))
        t += step
    return out


class AssignWordsTest(unittest.TestCase):
    def test_word_without_any_turn_is_none_never_s1(self):
        words = [W("你", 0.0, 0.2), W("好", 5.0, 5.2)]
        out = sp.assign_words(words, [T(0.0, 1.0, "S1")])
        self.assertEqual(out[0]["speaker"], "S1")
        self.assertIsNone(out[1]["speaker"])
        self.assertIn("no_turn", out[1]["reasons"])

    def test_no_turns_at_all_gives_none_everywhere(self):
        out = sp.assign_words(chars("你好吗", 0.0), [])
        self.assertTrue(all(w["speaker"] is None for w in out))

    def test_low_overlap_ratio_is_none(self):
        # chữ 1.0–1.2, turn chỉ phủ 0.05 s (25 %) → dưới ngưỡng 0,5 (tắt luật mép)
        out = sp.assign_words([W("好", 1.0, 1.2)], [T(0.0, 1.05, "S1")], edge_collar=None)
        self.assertIsNone(out[0]["speaker"])
        self.assertIn("low_ratio", out[0]["reasons"])
        # bật luật mép nhưng có người khác sát bên → vẫn None
        out = sp.assign_words([W("好", 1.0, 1.2)], [T(0.0, 1.05, "S1"), T(1.35, 2.0, "S2")])
        self.assertIsNone(out[0]["speaker"])

    def test_stretched_edge_word_with_single_speaker_is_assigned_as_edge(self):
        # chữ cuối câu bị kéo dài 1,1 s vào khoảng lặng, chỉ S1 phủ 0,3 s → gán S1 ("edge")
        words = [W("我", 0.0, 0.2), W("是", 0.2, 1.3)]
        turns = [T(0.0, 0.5, "S1"), T(3.0, 4.0, "S2")]
        out = sp.assign_words(words, turns)
        self.assertEqual(out[1]["speaker"], "S1")
        self.assertIn("edge", out[1]["reasons"])
        strict = sp.assign_words(words, turns, edge_collar=None)      # đúng nguyên văn ngưỡng 0,5
        self.assertIsNone(strict[1]["speaker"])

    def test_boundary_word_near_another_speaker_stays_none(self):
        # chữ ở ranh giới: S2 bắt đầu trong chữ, S1 vừa kết thúc sát trước → mơ hồ → None
        out = sp.assign_words([W("你", 1.0, 1.1)], [T(0.0, 0.98, "S1"), T(1.07, 3.0, "S2")])
        self.assertIsNone(out[0]["speaker"])
        self.assertIn("low_ratio", out[0]["reasons"])

    def test_boundary_word_after_comma_joins_following_clause(self):
        # "…他，[你]跟着…": "你" sau dấu phẩy, S2 bắt đầu giữa chữ → theo mệnh đề sau (S2)
        words = [W("他", 0.5, 0.95), W("，", 0.95, 1.0), W("你", 1.0, 1.1), W("跟", 1.1, 1.3), W("着", 1.3, 1.5)]
        out = sp.assign_words(words, [T(0.0, 0.98, "S1"), T(1.07, 3.0, "S2")])
        self.assertEqual([w["speaker"] for w in out], ["S1", "S1", "S2", "S2", "S2"])
        self.assertIn("edge", out[2]["reasons"])
        strict = sp.assign_words(words, [T(0.0, 0.98, "S1"), T(1.07, 3.0, "S2")], edge_collar=None)
        self.assertIsNone(strict[2]["speaker"])

    def test_word_between_two_punctuations_stays_none(self):
        words = [W("他", 0.5, 0.95), W("，", 0.95, 1.0), W("你", 1.0, 1.1), W("。", 1.1, 1.1), W("跟", 1.2, 1.4)]
        out = sp.assign_words(words, [T(0.0, 0.98, "S1"), T(1.07, 3.0, "S2")])
        self.assertIsNone(out[2]["speaker"])

    def test_coverage_ignores_time_given_to_punctuation_rows(self):
        words = [W("你", 0.0, 0.2), W("，", 0.2, 0.6), W("好", 0.6, 0.8)]
        seg = sp.build_segments(sp.assign_words(words, [T(0.0, 1.0, "S1")]), [T(0.0, 1.0, "S1")])[0]
        self.assertEqual(seg["coverage"], 1.0)

    def test_edge_segment_can_be_excluded_from_certain(self):
        words = chars("我是警察", 0.0, 0.3)
        words[-1] = W("察", 0.9, 2.2)
        turns = [T(0.0, 1.3, "S1")]
        seg = sp.build_segments(sp.assign_words(words, turns), turns)[0]
        self.assertIn("edge", seg["reasons"])
        self.assertFalse(sp.is_certain(seg, sp.CertainRule(allow_edge=False)))

    def test_picks_speaker_with_largest_overlap(self):
        out = sp.assign_words([W("好", 1.0, 1.4)], [T(0.0, 1.1, "S1"), T(1.1, 2.0, "S2")])
        self.assertEqual(out[0]["speaker"], "S2")
        self.assertAlmostEqual(out[0]["ratio"], 0.75, places=3)

    def test_exclusive_turns_take_precedence(self):
        turns = [T(0.0, 2.0, "S1"), T(0.5, 2.0, "S2")]
        exclusive = [T(0.0, 0.5, "S1"), T(0.5, 2.0, "S2")]
        overlaps = sp.turns_overlaps(turns)
        out = sp.assign_words([W("好", 1.0, 1.2)], turns, exclusive, overlaps)
        self.assertEqual(out[0]["speaker"], "S2")
        self.assertAlmostEqual(out[0]["overlap_frac"], 1.0)

    def test_zero_length_word(self):
        turns = [T(0.0, 1.0, "S1"), T(3.0, 4.0, "S2")]
        out = sp.assign_words([W("啊", 1.2, 1.2), W("嗯", 2.0, 2.0), W("哦", 3.5, 3.5)], turns)
        self.assertEqual(out[0]["speaker"], "S1")      # cách 0,2 s ≤ 0,3
        self.assertIsNone(out[1]["speaker"])           # cách 1,0 s
        self.assertEqual(out[2]["speaker"], "S2")      # nằm trong turn

    def test_punctuation_takes_previous_word_speaker(self):
        words = [W("好", 0.0, 0.2), W("，", 0.2, 0.3), W("走", 1.0, 1.2), W("。", 1.2, 1.3)]
        out = sp.assign_words(words, [T(0.0, 0.25, "S1"), T(0.25, 2.0, "S2")])
        self.assertEqual([w["speaker"] for w in out], ["S1", "S1", "S2", "S2"])
        self.assertTrue(out[1]["punct"])

    def test_leading_punctuation_has_no_speaker(self):
        out = sp.assign_words([W("“", 0.0, 0.0), W("好", 0.0, 0.2)], [T(0.0, 1.0, "S1")])
        self.assertIsNone(out[0]["speaker"])
        self.assertEqual(out[1]["speaker"], "S1")


class SmoothRunsTest(unittest.TestCase):
    def test_flicker_inside_sentence_is_smoothed_and_marked(self):
        words = chars("我今天去了学校", 0.0, 0.1)
        turns = [T(0.0, 0.3, "S1"), T(0.3, 0.4, "S2"), T(0.4, 1.0, "S1")]
        out = sp.smooth_runs(sp.assign_words(words, turns))
        self.assertTrue(all(w["speaker"] == "S1" for w in out))
        self.assertEqual(out[3]["reasons"], ["smoothed"])
        self.assertEqual(out[3]["smoothed_from"], "S2")

    def test_short_real_interjection_across_sentence_end_is_kept(self):
        words = chars("你走吧", 0.0, 0.2) + [W("。", 0.6, 0.6)] + [W("好", 0.7, 0.9), W("。", 0.9, 0.9)] \
            + chars("我也走", 1.0, 0.2)
        turns = [T(0.0, 0.65, "S1"), T(0.65, 0.95, "S2"), T(0.95, 2.0, "S1")]
        out = sp.smooth_runs(sp.assign_words(words, turns))
        self.assertEqual(out[4]["speaker"], "S2")
        self.assertNotIn("smoothed", out[4]["reasons"])

    def test_long_run_is_not_smoothed(self):
        words = chars("我今天去了学校", 0.0, 0.2)
        turns = [T(0.0, 0.4, "S1"), T(0.4, 1.0, "S2"), T(1.0, 2.0, "S1")]
        out = sp.smooth_runs(sp.assign_words(words, turns))
        self.assertEqual([w["speaker"] for w in out], ["S1", "S1", "S2", "S2", "S2", "S1", "S1"])

    def test_none_run_between_same_speaker_is_smoothed_not_defaulted(self):
        words = chars("我今天去", 0.0, 0.1)
        turns = [T(0.0, 0.1, "S1"), T(0.2, 0.4, "S1")]   # chữ thứ 2 không có turn
        out = sp.smooth_runs(sp.assign_words(words, turns))
        self.assertEqual(out[1]["speaker"], "S1")
        self.assertIn("smoothed", out[1]["reasons"])


class BuildSegmentsTest(unittest.TestCase):
    def test_speaker_change_mid_sentence_without_punctuation_splits(self):
        words = chars("你好我是警察", 0.0, 0.2)
        turns = [T(0.0, 0.4, "S1"), T(0.4, 1.2, "S2")]
        segs = sp.build_segments(sp.assign_words(words, turns), turns)
        self.assertEqual([(s["speaker"], s["text"]) for s in segs], [("S1", "你好"), ("S2", "我是警察")])
        self.assertEqual((segs[1]["word_start"], segs[1]["word_end"]), (2, 5))

    def test_punctuation_at_speaker_boundary_stays_with_previous(self):
        words = chars("你好", 0.0, 0.3) + [W("，", 0.6, 0.6)] + chars("我是", 0.6, 0.3)
        turns = [T(0.0, 0.6, "S1"), T(0.6, 1.2, "S2")]
        segs = sp.build_segments(sp.assign_words(words, turns), turns)
        self.assertEqual([s["text"] for s in segs], ["你好，", "我是"])

    def test_sentence_end_pause_and_max_len_split(self):
        words = chars("你好", 0.0) + [W("。", 0.4, 0.4)] + chars("走吧", 0.5) + chars("快点", 2.0)
        turns = [T(0.0, 3.0, "S1")]
        segs = sp.build_segments(sp.assign_words(words, turns), turns)
        self.assertEqual([s["text"] for s in segs], ["你好。", "走吧", "快点"])
        long_words = chars("一" * 100, 0.0, 0.2)   # 20 s liền
        segs = sp.build_segments(sp.assign_words(long_words, [T(0, 30, "S1")]), [T(0, 30, "S1")])
        self.assertTrue(all(s["end"] - s["start"] <= 15.0 for s in segs))
        self.assertEqual(len(segs), 2)

    def test_clean_sentences_have_purity_one_and_are_certain(self):
        words = chars("你好吗", 0.0, 0.3) + [W("？", 0.9, 0.9)] + chars("我很好", 1.0, 0.3)
        turns = [T(0.0, 0.95, "S1"), T(0.95, 2.0, "S2")]
        segs = sp.build_segments(sp.assign_words(words, turns), turns)
        self.assertEqual([s["purity"] for s in segs], [1.0, 1.0])
        self.assertTrue(all(s["certain"] and s["status"] == "assigned" for s in segs))

    def test_unassigned_words_form_their_own_uncertain_segment(self):
        words = chars("你好", 0.0, 0.3) + chars("谁啊", 0.6, 0.3)
        turns = [T(0.0, 0.6, "S1")]
        segs = sp.build_segments(sp.assign_words(words, turns), turns)
        self.assertEqual([s["speaker"] for s in segs], ["S1", None])
        self.assertFalse(segs[1]["certain"])
        self.assertEqual(segs[1]["status"], "no_turn")
        self.assertEqual(segs[1]["confidence"], 0.0)

    def test_overlap_and_short_segments_are_not_certain(self):
        words = chars("你好吗朋友", 0.0, 0.3)
        turns = [T(0.0, 1.5, "S1"), T(0.0, 1.5, "S2")]
        exclusive = [T(0.0, 1.5, "S1")]
        out = sp.assign_words(words, turns, exclusive, sp.turns_overlaps(turns))
        seg = sp.build_segments(out, turns)[0]
        self.assertEqual(seg["status"], "overlap")
        self.assertFalse(seg["certain"])
        short = sp.build_segments(sp.assign_words([W("好", 0, 0.3)], [T(0, 1, "S1")]), [T(0, 1, "S1")])[0]
        self.assertEqual(short["status"], "short")
        self.assertFalse(short["certain"])

    def test_smoothed_segment_is_not_certain_by_default(self):
        words = chars("我今天去了学校", 0.0, 0.1)
        turns = [T(0.0, 0.3, "S1"), T(0.3, 0.4, "S2"), T(0.4, 1.0, "S1")]
        seg = sp.build_segments(sp.smooth_runs(sp.assign_words(words, turns)), turns)[0]
        self.assertIn("smoothed", seg["reasons"])
        self.assertFalse(seg["certain"])
        self.assertTrue(sp.is_certain(seg, sp.CertainRule(allow_smoothed=True, min_coverage=0.5)))


class MatchingTest(unittest.TestCase):
    def test_hungarian_matches_brute_force(self):
        import random
        rnd = random.Random(7)
        for n, m in [(3, 3), (4, 6), (6, 4), (5, 5)]:
            cost = [[rnd.randint(0, 30) for _ in range(m)] for _ in range(n)]
            rows, cols = sp.linear_sum_assignment(cost)
            got = sum(cost[r][c] for r, c in zip(rows, cols))
            k = min(n, m)
            if n <= m:
                best = min(sum(cost[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
            else:
                best = min(sum(cost[p[j]][j] for j in range(m)) for p in itertools.permutations(range(n), m))
            self.assertEqual(len(rows), k)
            self.assertEqual(got, best)

    def test_label_map_and_disagreements(self):
        words = chars("你好我是警察", 0.0, 0.2)
        a = sp.assign_words(words, [T(0.0, 0.4, "S1"), T(0.4, 1.2, "S2")])
        b_same = sp.assign_words(words, [T(0.0, 0.4, "S7"), T(0.4, 1.2, "S3")])
        b_diff = sp.assign_words(words, [T(0.0, 1.2, "S1")])
        self.assertEqual(sp.label_map(a, b_same), {"S7": "S1", "S3": "S2"})
        segs = sp.build_segments(a, None)
        sp.disagreements(segs, a, {"same": b_same, "diff": b_diff})
        # "diff" gộp hai người làm một: Hungarian ghép nhãn đó với S2 (chồng lấn dài hơn) nên
        # đoạn của S1 là đoạn bất đồng.
        self.assertEqual(segs[0]["disagree_arms"], ["diff"])
        self.assertEqual(segs[1]["disagree_arms"], [])

    def test_merge_suggestions(self):
        emb = {"S1": [[1.0, 0.0], [0.9, 0.1]], "S2": [0.95, 0.05], "S3": [0.0, 1.0]}
        got = sp.merge_suggestions(emb, threshold=0.75)
        self.assertEqual([(g["a"], g["b"]) for g in got], [("S1", "S2")])


class TurnsTest(unittest.TestCase):
    def test_normalize_labels_by_first_appearance(self):
        turns, (exclusive,), mapping = sp.normalize_labels(
            [T(5, 6, "SPEAKER_03"), T(0, 1, "SPEAKER_07")], [T(0, 1, "SPEAKER_07")])
        self.assertEqual(mapping, {"SPEAKER_07": "S1", "SPEAKER_03": "S2"})
        self.assertEqual([t["speaker"] for t in turns], ["S1", "S2"])
        self.assertEqual(exclusive[0]["speaker"], "S1")

    def test_turns_overlaps(self):
        got = sp.turns_overlaps([T(0, 2, "S1"), T(1, 3, "S2"), T(1.5, 2.5, "S1"), T(4, 5, "S1")])
        self.assertEqual(got, [{"start": 1, "end": 2.5, "speakers": ["S1", "S2"]}])


class ResolveJudgedTest(unittest.TestCase):
    LABELS = ["S1", "S2", "S3", "S4"]

    def seg(self, speaker="S2", certain=True):
        return {"id": "seg_0001", "speaker": speaker, "certain": certain}

    def flag(self, flagged=True, voters=None):
        return {"flag": flagged, "voters": voters or {"3dspeaker": "S2", "sherpa-eres2netv2": "S2"}}

    @staticmethod
    def vis(speaker, conf=0.9, split_at=None):
        return {"final": {"speaker": speaker, "confidence": conf, "split_at": split_at}}

    def resolve(self, seg, flag, text, vision, **kw):
        return sp.resolve_judged(seg, flag, text, vision, labels=self.LABELS, **kw)

    def test_unflagged_segment_is_decided_by_machine_without_judge(self):
        got = self.resolve(self.seg(), self.flag(False), None, None)
        self.assertEqual((got["speaker"], got["certain"], got["source"]), ("S2", True, "machine"))

    def test_llm_and_vlm_agree_confidently_resolves_flagged_segment(self):
        got = self.resolve(self.seg("S2", certain=False), self.flag(), {"label": "S2"}, self.vis("S2"))
        self.assertEqual((got["speaker"], got["certain"], got["source"]), ("S2", True, "judge"))
        self.assertEqual(got["why"], ["llm_vlm_agree"])

    def test_agreement_can_change_label_only_to_an_acoustically_plausible_voice(self):
        close = [("S2", "S4")]
        got = self.resolve(self.seg("S2"), self.flag(), {"label": "S4"}, self.vis("S4"), close_pairs=close)
        self.assertEqual((got["speaker"], got["source"], got["why"]), ("S4", "judge", ["llm_vlm_agree", "changed"]))
        # S3 không phải nhãn máy, không phải arm bỏ phiếu, không cùng cặp giọng → người duyệt
        got = self.resolve(self.seg("S2"), self.flag(), {"label": "S3"}, self.vis("S3"), close_pairs=close)
        self.assertEqual((got["source"], got["certain"], got["speaker"]), ("review", False, "S2"))
        self.assertIn("implausible", got["why"])
        # arm bỏ phiếu nói S3 → hợp lý
        got = self.resolve(self.seg("S2"), self.flag(voters={"3dspeaker": "S3"}), {"label": "S3"}, self.vis("S3"))
        self.assertEqual((got["speaker"], got["source"]), ("S3", "judge"))

    def test_single_source_never_assigns(self):
        for text, vision in (({"label": "S2"}, None), (None, self.vis("S2")), ({"label": "S2"}, self.vis("S4"))):
            got = self.resolve(self.seg("S2", certain=False), self.flag(), text, vision)
            self.assertEqual(got["source"], "review")
            self.assertFalse(got["certain"])

    def test_low_vision_confidence_goes_to_review_with_suggestion(self):
        got = self.resolve(self.seg("S2"), self.flag(), {"label": "S2"}, self.vis("S2", conf=0.6))
        self.assertEqual((got["source"], got["suggestion"], got["why"]), ("review", "S2", ["vlm_low_conf"]))

    def test_multiple_speakers_suggests_split(self):
        got = self.resolve(self.seg("S4"), self.flag(), {"label": "S4"}, self.vis("multiple", split_at=277.2))
        self.assertEqual((got["source"], got["split_at"]), ("review", 277.2))
        self.assertIn("multiple", got["why"])

    def test_new_character_and_offscreen_go_to_review(self):
        got = self.resolve(self.seg("S1"), self.flag(), {"label": "NEW"}, self.vis("offscreen"))
        self.assertEqual(got["source"], "review")
        self.assertEqual(set(got["why"]), {"llm_new", "vlm_offscreen"})

    def test_machine_without_label_accepts_agreement(self):
        got = self.resolve(self.seg(None, certain=False), self.flag(voters={}), {"label": "S3"}, self.vis("S3", 0.85))
        self.assertEqual((got["speaker"], got["certain"], got["source"]), ("S3", True, "judge"))

    # --- đoạn nhiều người (mục 6.8b): phần đã khớp vào ranh giới chữ ---------------------------------

    @staticmethod
    def parts(*spec):
        return [{"w0": w0, "speaker": spk, "confidence": conf, "start": 10.0 + w0 * 0.1} for w0, spk, conf in spec]

    def split_vis(self, *spec):
        return {"final": {"speaker": "multiple", "confidence": min(c for _, _, c in spec), "parts": self.parts(*spec)}}

    def test_same_split_from_llm_and_vlm_is_decided_automatically(self):
        flag = dict(self.flag(), inner_speakers=["S3", "S4"])
        text = {"label": "MULTI", "parts": self.parts((0, "S3", None), (8, "S4", None))}
        got = self.resolve(self.seg("S3"), flag, text, self.split_vis((0, "S3", 0.9), (9, "S4", 0.85)))
        self.assertEqual((got["source"], got["certain"], got["speaker"], got["why"]), ("judge", True, "S3", ["llm_vlm_agree", "split"]))
        self.assertEqual([(p["w0"], p["speaker"]) for p in got["parts"]], [(0, "S3"), (9, "S4")])

    def test_split_needs_both_sources_same_cut_same_voices_and_confidence(self):
        flag = dict(self.flag(), inner_speakers=["S3", "S4"])
        v = self.split_vis((0, "S3", 0.9), (9, "S4", 0.9))
        cases = [
            ({"label": "S3", "parts": self.parts((0, "S3", None))}, v, "split_disagree"),             # LLM: một người
            ({"parts": self.parts((0, "S3", None), (4, "S4", None))}, v, "split_disagree"),          # cắt lệch 5 chữ
            ({"parts": self.parts((0, "S3", None), (9, "S2", None))}, v, "split_disagree"),          # khác giọng
            ({"parts": self.parts((0, "S3", None), (9, "S4", None))}, self.split_vis((0, "S3", 0.9), (9, "S4", 0.6)), "vlm_low_conf"),
            ({"parts": self.parts((0, "S3", None), (9, "S1", None))}, self.split_vis((0, "S3", 0.9), (9, "S1", 0.9)), "implausible"),
        ]
        for text, vision, reason in cases:
            got = self.resolve(self.seg("S3"), flag, text, vision)
            self.assertEqual((got["source"], got["certain"], got["speaker"]), ("review", False, "S3"), reason)
            self.assertIn(reason, got["why"])
            self.assertEqual(len(got["parts"]), 2)          # gợi ý tách cho người duyệt
        # chỉ LLM tách → gợi ý là cách chia của LLM
        got = self.resolve(self.seg("S3"), flag, {"parts": self.parts((0, "S3", None), (9, "S4", None))}, self.vis("S3"))
        self.assertEqual((got["source"], got["split_at"]), ("review", 10.9))

    def test_voice_heard_inside_segment_makes_single_label_plausible(self):
        flag = dict(self.flag(voters={"3dspeaker": "S2"}), inner_speakers=["S2", "S1"])
        got = self.resolve(self.seg("S2"), flag, {"label": "S1"}, self.vis("S1"))
        self.assertEqual((got["speaker"], got["source"]), ("S1", "judge"))

    # --- phiếu MẶT (faces.py, thí nghiệm E1) ---------------------------------------------------------

    @staticmethod
    def face(speaker=None, conf=0.9, vote=None, parts=None):
        return {"vote": vote or ("multiple" if parts else "speaker"), "speaker": speaker, "confidence": conf,
                "parts": parts}

    @staticmethod
    def fparts(*spec):
        return [{"w0": w0, "speaker": spk, "words": n, "start": 10.0 + w0 * 0.1} for w0, spk, n in spec]

    def test_face_without_disagreement_changes_nothing(self):
        got = self.resolve(self.seg(), self.flag(False), None, None, face=self.face("S2"))
        self.assertEqual((got["source"], got["speaker"]), ("machine", "S2"))
        got = self.resolve(self.seg("S2", certain=False), self.flag(), {"label": "S2"}, self.vis("S2"),
                           face=self.face(None, 1.0, vote="offscreen"))
        self.assertEqual((got["source"], got["why"]), ("judge", ["llm_vlm_agree"]))

    def test_face_vetoes_machine_and_judge_decisions(self):
        got = self.resolve(self.seg("S2"), self.flag(False), None, None, face=self.face("S4"))
        self.assertEqual((got["source"], got["certain"], got["speaker"], got["why"]), ("review", False, "S2", ["face_disagree"]))
        got = self.resolve(self.seg("S2"), self.flag(), {"label": "S2"}, self.vis("S2"), face=self.face("S4"))
        self.assertEqual((got["source"], got["suggestion"]), ("review", "S2"))
        self.assertEqual(got["why"], ["face_veto"])
        # MẶT chưa đủ tin → không phủ quyết
        got = self.resolve(self.seg("S2"), self.flag(), {"label": "S2"}, self.vis("S2"), face=self.face("S4", conf=0.3))
        self.assertEqual(got["source"], "judge")

    def test_face_part_by_another_voice_sends_segment_to_review(self):
        fp = self.fparts((0, "S2", 5), (6, "NEW_FACE:F5", 7))
        got = self.resolve(self.seg("S4"), self.flag(), {"label": "S4"}, self.vis("S4"), face=self.face(parts=fp))
        self.assertEqual((got["source"], got["why"]), ("review", ["face_multiple"]))
        self.assertIsNone(got["parts"])        # phần "mặt mới" không phải giọng có thật → không gợi ý tách
        # phần giọng khác quá ngắn, hoặc chỉ là mặt chưa gắn giọng → giữ quyết định
        for fp in (self.fparts((0, "S2", 2), (3, "S4", 9)), self.fparts((0, "S4", 9), (10, "NEW_FACE:F12", 6))):
            got = self.resolve(self.seg("S4"), self.flag(False), None, None, face=self.face(parts=fp))
            self.assertEqual(got["source"], "machine")

    def test_llm_and_face_agreeing_decide_without_vision(self):
        got = self.resolve(self.seg("S2", certain=False), self.flag(), {"label": "S4"}, None,
                           face=self.face("S4"), close_pairs=[("S2", "S4")])
        self.assertEqual((got["source"], got["speaker"], got["why"]), ("judge", "S4", ["llm_face_agree", "changed"]))
        # VLM đã chạy và nói khác → không đi tắt; phải hợp lý âm thanh; LLM tách → không đi tắt
        got = self.resolve(self.seg("S2"), self.flag(), {"label": "S4"}, self.vis("S3"), face=self.face("S4"),
                           close_pairs=[("S2", "S4")])
        self.assertEqual(got["source"], "review")
        got = self.resolve(self.seg("S2"), self.flag(), {"label": "S3"}, None, face=self.face("S3"))
        self.assertEqual(got["source"], "review")
        text = {"label": "MULTI", "parts": self.parts((0, "S4", None), (5, "S2", None))}
        got = self.resolve(self.seg("S2"), self.flag(), text, None, face=self.face("S2"))
        self.assertEqual(got["source"], "review")

    def test_face_split_is_offered_to_reviewer_only_with_real_voices(self):
        fp = self.fparts((0, "S2", 7), (8, "S4", 4))
        got = self.resolve(self.seg("S2"), self.flag(), {"label": "S4"}, self.vis("S2", conf=0.6), face=self.face(parts=fp))
        self.assertEqual((got["source"], got["split_at"]), ("review", 10.8))
        self.assertIn("face_split", got["why"])
        self.assertEqual([p["speaker"] for p in got["parts"]], ["S2", "S4"])


if __name__ == "__main__":
    unittest.main()
