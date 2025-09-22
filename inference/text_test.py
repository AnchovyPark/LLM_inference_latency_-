# --- 설정 ---
# 1. 읽어올 텍스트 파일 경로
file_path = "/Users/anchovy-mac/Desktop/calculating/inference/input_text.txt"  # 실제 파일명으로 수정해주세요.

# 2. 잘라낼 리스트의 길이
target_length = 100
# ------------

try:
    # 1. 텍스트 파일을 읽어 하나의 문자열(str)로 변환
    print(f"'{file_path}' 파일을 읽어 문자열로 변환합니다...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text_content_str = f.read()
    print("-> 문자열 변환 완료!")

    # 2. 문자열을 단어 리스트(list)로 변환
    print("\n문자열을 단어 리스트로 변환합니다...")
    word_list = text_content_str.split()  # 공백(띄어쓰기, 줄바꿈 등) 기준으로 자름
    print("-> 리스트 변환 완료!")

    # 3. 리스트를 원하는 길이로 슬라이싱
    print(f"\n리스트를 '{target_length}' 길이로 슬라이싱합니다...")
    sliced_list = word_list[:target_length]
    print("-> 슬라이싱 완료!")

    # --- 최종 결과 확인 ---
    print("\n--- 결과 ---")
    print(f"원본 글자 수: {len(text_content_str)}")
    print(f"리스트의 총 단어 수: {len(word_list)}")
    print(f"슬라이싱 후 단어 수: {len(sliced_list)}")
    print(f"\n슬라이싱된 리스트의 처음 10개 단어:")
    print(sliced_list[:10])
    print("\n슬라이싱된 리스트의 마지막 10개 단어:")
    print(sliced_list[-10:])
    print("\n--- 테스트 성공 ---")


except FileNotFoundError:
    print(f"\n[오류] '{file_path}' 파일을 찾을 수 없습니다. 파일명을 확인해주세요.")
except Exception as e:
    print(f"\n[오류] 예상치 못한 오류 발생: {e}")