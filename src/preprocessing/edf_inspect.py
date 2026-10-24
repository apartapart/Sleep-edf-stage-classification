"""EDF 파일 헤더/채널/샘플 수 확인 (pyedflib)."""
import pyedflib


def inspect_edf(psg_file: str) -> None:
    """PSG EDF의 채널 수, 샘플 수, 헤더, 채널별 헤더를 출력."""
    try:
        with pyedflib.EdfReader(psg_file) as f:
            num_signals = f.signals_in_file
            signals = [f.readSignal(i) for i in range(num_signals)]

            header = f.getHeader()
            signal_headers = f.getSignalHeaders()

            num_channels = len(signals)
            print(f"총 채널 수: {num_channels}")

            if num_channels > 0:
                num_samples = len(signals[0])
                print(f"첫 번째 채널 샘플 수: {num_samples}")

                print("\nEDF 파일 헤더 정보:")
                print(header)

                print("\n각 채널 헤더 정보:")
                for i, h in enumerate(signal_headers):
                    print(f"채널 {i+1}: {h['label']} | 샘플 수: {len(signals[i])} | 단위: {h['dimension']}")

    except FileNotFoundError:
        print(f"Error: The file {psg_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
