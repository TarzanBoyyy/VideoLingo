from core.prompts import generate_shared_prompt, get_prompt_faithfulness, get_prompt_direct_and_free
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *
console = Console()

def valid_translate_result(result: dict, required_keys: list, required_sub_keys: list):
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {"status": "error", "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"}
    
    # Check for required sub-keys in all items
    for key in result:
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {"status": "error", "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"}

    return {"status": "success", "message": "Translation completed"}

def translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt, index = 0):
    check_cancel()
    shared_prompt = generate_shared_prompt(previous_content_prompt, after_cotent_prompt, summary_prompt, things_to_note_prompt)

    # Retry translation if the length of the original text and the translated text are not the same, or if the specified key is missing
    def retry_translation(prompt, length, step_name, sub_keys):
        def valid(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], sub_keys)
        for retry in range(3):
            result = ask_gpt(prompt+retry* " ", resp_type='json', valid_def=valid, log_title=f'translate_{step_name}')
            if len(lines.split('\n')) == len(result):
                return result
            if retry != 2:
                console.print(f'[yellow]⚠️ {step_name.capitalize()} translation of block {index} failed, Retry...[/yellow]')
        raise ValueError(f'[red]❌ {step_name.capitalize()} translation of block {index} failed after 3 retries. Please check `output/gpt_log/error.json` for more details.[/red]')

    n_lines = len(lines.split('\n'))
    origin_lines = lines.split('\n')

    # reflect_translate: merged ONE call emitting both `direct` and `free` per line —
    # `free` is generated after `direct` in the same response, keeping the
    # reflect-and-refine behavior while halving requests and cutting ~40% of the
    # output tokens (no `origin` echo, no second call regenerating direct).
    if load_key('reflect_translate'):
        merged = retry_translation(get_prompt_direct_and_free(lines, shared_prompt), n_lines, 'merged', ['direct', 'free'])
        for key in merged:
            merged[key]["direct"] = merged[key]["direct"].replace('\n', ' ')
            merged[key]["free"] = merged[key]["free"].replace('\n', ' ')
        table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        for i, key in enumerate(merged):
            table.add_row(f"[cyan]Origin:  {origin_lines[i]}[/cyan]")
            table.add_row(f"[magenta]Direct:  {merged[key]['direct']}[/magenta]")
            table.add_row(f"[green]Free:    {merged[key]['free']}[/green]")
            if i < len(merged) - 1:
                table.add_row("[yellow]" + "-" * 50 + "[/yellow]")
        console.print(table)
        translate_result = "\n".join([merged[i]["free"].strip() for i in merged])
        if len(translate_result.split('\n')) != n_lines:
            raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')
        return translate_result, lines

    ## Single-pass faithful translation (reflect_translate off)
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, n_lines, 'faithfulness', ['direct'])

    for i in faith_result:
        faith_result[i]["direct"] = faith_result[i]["direct"].replace('\n', ' ')

    translate_result = "\n".join([faith_result[i]["direct"].strip() for i in faith_result])

    table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
    table.add_column("Translations", style="bold")
    for i, key in enumerate(faith_result):
        table.add_row(f"[cyan]Origin:  {origin_lines[i]}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        if i < len(faith_result) - 1:
            table.add_row("[yellow]" + "-" * 50 + "[/yellow]")

    console.print(table)
    return translate_result, lines


if __name__ == '__main__':
    # test e.g.
    lines = '''All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain.'''
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt)