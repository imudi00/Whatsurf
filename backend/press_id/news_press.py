import requests             # 웹사이트에 접속해서 HTML 데이터 가져옴
from bs4 import BeautifulSoup # 가져온 HTML에서 원하는 정보 골라내는 라이브러리
import re                   # 'oid=숫자' 같은 특정 패턴(정규표현식)을 찾기 위한 도구
import csv                  # CSV 파일로 저장하는 도구

def save_naver_press_to_csv():
    # 네이버 뉴스 언론사 목록 페이지 접속
    url = "https://news.naver.com/main/officeList.naver"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        # 브라우저인 척 속이기 (차단 방지용)
    }
    
    try:
        response = requests.get(url, headers=headers) # 해당 URL에 접속해서 데이터 받기
        response.raise_for_status() # 접속 에러 발생 시 중단
    except Exception as e:
        print(f"페이지 접속 중 오류가 발생했습니다: {e}")
        return
    
    # 받아온 HTML 글자들을 파이썬이 이해할 수 있는 구조로 변환
    soup = BeautifulSoup(response.text, 'lxml')
    
    #언론사 데이터 추출 (중복 제거를 위해 set 사용)
    press_data = []
    seen_oids = set()
    
    # 'oid='가 포함된 모든 링크 찾기
    links = soup.find_all('a', href=re.compile(r'oid=(\d+)'))
    
    for link in links:
        name = link.get_text().strip()
        href = link['href']
        
        # oid 값만 추출
        match = re.search(r'oid=(\d+)', href)
        if match:
            oid = match.group(1)
            # 이름이 있고, 아직 추가되지 않은 oid만 리스트에 넣음
            if name and oid not in seen_oids:
                press_data.append([name, oid])
                seen_oids.add(oid)

    #CSV 파일로 저장
    filename = "naver_press_ids.csv"
    try:
        # utf-8-sig로 저장 -> 한글 깨짐 방지
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # 칼럼명 작성
            writer.writerow(['언론사명', 'oid'])
            # 데이터 작성
            writer.writerows(press_data)
            
        print(f"총 {len(press_data)}개 '{filename}'으로 저장")
    except Exception as e:
        print(f"파일 저장 중 오류가 발생: {e}")

# 실행
if __name__ == "__main__":
    save_naver_press_to_csv()