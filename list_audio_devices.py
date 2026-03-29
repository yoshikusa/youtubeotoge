"""
sounddevice が見える入力デバイス一覧（番号 = audio_settings.json の input_device_index）。

使い方:
  python list_audio_devices.py
"""

from __future__ import annotations


def main() -> None:
    import sounddevice as sd

    try:
        default = sd.default.device
    except Exception:
        default = None
    def_inp = None
    if default is not None:
        inp = getattr(default, "input", None)
        if inp is not None:
            def_inp = int(inp)
        elif isinstance(default, (tuple, list)):
            def_inp = int(default[0])
        else:
            try:
                def_inp = int(default[0])
            except (TypeError, IndexError, ValueError):
                pass

    print("Index | In ch | Name")
    print("------+-------+-----")
    for i, d in enumerate(sd.query_devices()):
        inch = int(d.get("max_input_channels", 0) or 0)
        name = str(d.get("name", ""))
        mark = "  (既定入力)" if def_inp is not None and i == def_inp else ""
        if inch <= 0:
            print(f"{i:5} |   —   | {name}{mark}")
        else:
            print(f"{i:5} | {inch:5} | {name}{mark}")
    print()
    print("入力を取るには In ch が 1 以上の行の Index を audio_settings.json に書きます。")


if __name__ == "__main__":
    main()
