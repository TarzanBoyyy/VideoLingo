from ruamel.yaml import YAML
import os
import threading

CONFIG_PATH = 'config.yaml'
lock = threading.Lock()

yaml = YAML()
yaml.preserve_quotes = True

# -----------------------
# load & update config
# -----------------------

# Dashboard (~/project/video-dashboard) đặt provider dịch qua env cho từng job, để đổi
# nhà cung cấp không phải sửa config.yaml giữa các job — cùng kiểu với VIENEU_VOICE
# của core/tts_backend/custom_tts.py và VL_KEEP_SRC_SUB của core/_7_sub_into_vid.py.
ENV_OVERRIDES = {
    "api.base_url": "VL_API_BASE_URL",
    "api.model": "VL_API_MODEL",
    "api.key": "VL_API_KEY",
}

def load_key(key):
    env_name = ENV_OVERRIDES.get(key)
    if env_name:
        value = os.environ.get(env_name)
        if value:
            return value

    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

    keys = key.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(f"Key '{k}' not found in configuration")
    return value

def update_key(key, new_value):
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

        keys = key.split('.')
        current = data
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False

        if isinstance(current, dict) and keys[-1] in current:
            current[keys[-1]] = new_value
            with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
                yaml.dump(data, file)
            return True
        else:
            raise KeyError(f"Key '{keys[-1]}' not found in configuration")
        
# basic utils
def get_joiner(language):
    if language in load_key('language_split_with_space'):
        return " "
    elif language in load_key('language_split_without_space'):
        return ""
    else:
        raise ValueError(f"Unsupported language code: {language}")

if __name__ == "__main__":
    print(load_key('language_split_with_space'))
