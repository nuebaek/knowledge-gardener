# 🗓️ week-02-TIL — HTTP, LLM 연동, DB, 서비스 구조

> 브라우저의 요청이 AI 모델 응답으로 돌아오기까지, 백엔드 서비스의 전체 흐름을 한 층씩 이해한 한 주

**기간:** 2026-05-18 ~ 2026-05-22

**키워드:** `HTTP` `FastAPI` `Ollama` `httpx` `직렬화` `스트리밍` `예외처리` `Database` `SQL` `디자인패턴` `CORS` `HTTPS`

---

## 📋 이번 주 학습 지도

| 날짜 | 주제 | 핵심 개념 |
|------|------|-----------|
| 05/18 | 클라이언트-서버, HTTP | 소켓, HTTP 메시지, FastAPI, REST, Pydantic |
| 05/19 | 로컬 LLM, 비동기 통신 | Ollama, httpx, 직렬화, 스트리밍, 예외처리 |
| 05/20 | 데이터베이스 기초 | RDB, SQL, ERD, Index, Transaction, NoSQL |
| 05/21 | 구조 개선, 프론트엔드 | 디자인 패턴, Route-Controller-Model, 미들웨어, HTML/JS, CORS, Streamlit |
| 05/22 | 딥다이브 | HTTP vs HTTPS, TLS 핸드셰이크 |

---

## 📖 학습 내용

### Day 1 (05/18) — 클라이언트, 서버, HTTP

#### 1-1. 클라이언트 - 서버

> **클라이언트**는 서버에 필요한 데이터나 응답을 요청하고, **서버**는 해당 요청을 받아서 결과를 반환한다.

**사용 이유** \
여러 사용자가 동일한 데이터를 공유하고 사용할 수 있도록 하기 위해서다.  
데이터와 비즈니스 로직을 중앙에서 관리하면 유지보수와 동기화가 쉬워진다.
  
**핵심** 
- 역할 구분이지 위치 구분이 아니다
- **같은 컴퓨터라도 요청하면 클라이언트, 응답하면 서버가 될 수 있다**
- 브라우저(클라이언트)는 서버에 요청하고, 서버는 HTML/JSON 등을 응답한다
- API 서버가 외부 API를 호출할 때는 서버도 클라이언트 역할을 한다

**흐름** \
웹 서비스는 대부분 다음 흐름으로 동작한다.
```text
클라이언트 → 요청 → 서버 → 처리 → 응답 → 클라이언트
```
이 구조 위에서 HTTP, FastAPI, DB, LLM 연동 같은 기술들이 하나씩 추가된다.

**비유**
> 클라이언트 = 주문하는 손님  
> 서버 = 주문을 처리하는 바리스타

🔗 [MDN - 클라이언트-서버 개요](https://developer.mozilla.org/ko/docs/Learn/Server-side/First_steps/Client-Server_overview)

---

#### 1-2. 소켓(Socket) / 포트(Port) / localhost

> 소켓은 네트워크 통신 기능에 프로그래밍적으로 접근하기 위한 인터페이스다. 포트는 한 컴퓨터에서 여러 프로그램을 구분하는 번호이고, localhost는 자기 자신(이 컴퓨터)을 가리키는 주소다.

**사용 이유** \
두 컴퓨터가 데이터를 주고받으려면 연결 통로가 필요하다.  
소켓은 그 통로를 만들고 데이터를 송수신할 수 있게 해준다.

**핵심**
- 소켓이 서버보다 큰 개념이다. 서버는 소켓을 사용하는 것이지, 소켓 = 서버가 아니다.
- 소켓은 TCP/IP 네트워크 기능을 사용할 수 있게 해주는 인터페이스
- 서버는 특정 포트에 바인딩되어 요청을 기다린다

**비유** 
> 소켓 = 전화기 자체  
> 포트 = 내선 번호  
> localhost = 자기 자신의 전화번호
  
**코드**

```python
import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(1)

client_socket, addr = server_socket.accept()
client_socket.sendall("Hello".encode('utf-8'))
```

- 문자 → 바이트: `encode('utf-8')`
- 바이트 → 문자: `decode('utf-8')`

---

#### 1-3. HTTP (HyperText Transfer Protocol)

> 웹에서 클라이언트와 서버가 데이터를 주고받기 위한 애플리케이션 계층 프로토콜.

**사용 이유** \
TCP는 데이터를 신뢰성 있게 전달해주지만, 메시지 형식 자체는 정의하지 않는다.  
HTTP는 요청과 응답이 어떤 구조로 오갈지 표준화한다.

**HyperText란** \
링크를 통해 다른 정보로 이동할 수 있는 텍스트. 웹에서 문서들이 서로 연결되는 방식의 기반이다.
(HTML, CSS, JS, 이미지 등 다양한 미디어를 포함한 형식으로 전달된다)

**핵심**
- HTTP는 TCP 위에서 동작한다
- 요청(Request)과 응답(Response) 구조를 정의한다
- HTTP는 **Stateless(무상태)** 프로토콜이다
- 이전 요청 상태를 기억하지 않기 때문에 쿠키/세션/JWT 등이 필요하다

**흐름** 
```text
TCP = 데이터를 안전하게 전달
HTTP = 어떤 형식으로 전달할지 정의
FastAPI = HTTP 요청을 실제 함수와 연결
```
HTTPS에서는 중간에 TLS 암호화 계층이 추가된다.

**HTTP의 역할** \
HTTP는 브라우저와 서버가 서로 다른 언어와 환경에서도 같은 방식으로 통신할 수 있게 해준다.
```text
브라우저(JS)
↕ HTTP
FastAPI(Python)
```

**비유**  
> TCP = 배달망 (내용물이 뭔지 상관없이 신뢰성 있게 전달)  
> HTTP = 주문서 형식 (메서드, URL, payload, 상태코드라는 약속)  
> HTTP 메시지가 TCP 배달망을 타고 전달된다  

---

#### 1-4. HTTP Message

> HTTP에서 실제로 주고받는 데이터 단위. 정해진 형식(시작줄 + 헤더 + 빈줄 + 본문)으로 구성된다.

**사용 이유** \
메시지 형식이 정해져 있어야 서버가 요청을 해석하고, 클라이언트가 응답을 이해할 수 있다.
HTTP Message가 그 표준 형식이다.

**핵심**

| 구성 요소 | 설명 | 비유 |
|---|---|---|
| 시작줄 (Start Line) | 뭘 요청/응답하는지 | 편지 제목 |
| 헤더 (Headers) | 본문에 대한 메타 정보 | 봉투에 적힌 정보 |
| 빈 줄 (Empty Line) | 헤더 끝, 본문 시작 구분선 | 헤더가 끝났다는 신호 |
| 본문 (Body) | 실제 데이터 (없을 수도 있음) | 편지 내용 |

```text
요청: GET /posts HTTP/1.1
      ↑     ↑      ↑
    메서드   경로  HTTP버전

응답: HTTP/1.1 200 OK
                ↑
              상태코드
```

**비유**  
> 요청 메시지 = 주문서: `GET /menu HTTP/1.1 → Host: cafe.com → (빈줄) → (본문 없음)`  
> 응답 메시지 = 영수증+음식: `HTTP/1.1 200 OK → Content-Type: application/json → (빈줄) → {음식 데이터}`

---

#### 1-5. HTTP 상태 코드 (Status Code)

> 서버가 요청 처리 결과를 숫자로 알려주는 코드.

**사용 이유** \
클라이언트는 요청이 성공했는지, 실패했는지, 권한이 없는지 등을 알아야 한다.

**핵심**
- **2xx** = 성공 / **3xx** = 리다이렉트 / **4xx** = 클라이언트 잘못 / **5xx** = 서버 잘못
- `401` vs `403`: 401 = **"누구세요?"**(로그인 필요), 403 = **"알긴 아는데 안 돼"**(권한 없음)
- 상태 코드는 클라이언트와 서버 간 "결과 피드백" 역할

**대분류:**

| 번호대 | 설명 |
| --- | ----------------------- |
| 1xx | 정보 메시지 (Informational) |
| 2xx | 성공 (Successful) |
| 3xx | 리다이렉션 (Redirection) |
| 4xx | 클라이언트 오류 (Client Error) |
| 5xx | 서버 오류 (Server Error) |

**주요 상태 코드:**

| 코드 | 의미 | 한마디 |
| --- | --------------------- | ------------------------------- |
| 200 | OK | 요청 성공 |
| 201 | Created | 새 리소스 생성 성공 (POST 후) |
| 400 | Bad Request | 클라이언트가 요청을 잘못 보냄 |
| 401 | Unauthorized | 로그인이 필요함 |
| 403 | Forbidden | 로그인은 됐는데 권한이 없음 |
| 404 | Not Found | 그런 주소 없음 |
| 422 | Unprocessable Entity | 형식은 맞는데 내용이 이상함 (FastAPI 자주 등장) |
| 500 | Internal Server Error | 서버 내부 오류 |

---

#### 1-6. HTTP 요청 메서드 (Request Methods)

> HTTP 요청에서 클라이언트가 서버에게 "무엇을 하고 싶은지" 의도를 전달하는 동사.

**사용 이유** \
같은 URL이라도 데이터를 조회하는지, 생성하는지, 수정하는지 구분해야 하기 때문이다.

**핵심**
- **CRUD ↔ HTTP 메서드 매핑**: Create=POST / Read=GET / Update=PUT·PATCH / Delete=DELETE
- GET: 데이터 조회, 쿼리스트링으로 데이터 전달 (`GET /weather?city=seoul`)
- POST: 데이터 생성, Body에 데이터 첨부
- PUT: 데이터 **전체** 교체 (빠진 필드는 null/기본값)
- PATCH: 데이터 **일부** 수정
- DELETE: 데이터 삭제
- **멱등성(Idempotency)**: 같은 요청을 여러 번 해도 결과가 같은 것 (GET, PUT, DELETE = 멱등 / POST = 멱등 아님)

**비유** 
> GET = 메뉴 보기  
> POST = 주문하기  
> PUT = 주문 전체 바꾸기  
> PATCH = 주문 일부 수정  
> DELETE = 주문 취소  

---

#### 1-7. REST (RESTful API)

> API를 설계하는 원칙. URL은 자원(명사)으로, 행동은 HTTP 메서드(동사)로 표현하자는 약속.

**사용 이유** \
API를 일관성 있게 설계하기 위해서다.  
REST 원칙을 따르면 API만 봐도 역할을 예측하기 쉬워진다.
  
**핵심**
- **REST ≠ JSON.** REST는 **방식(설계 원칙)**, JSON은 **형식(데이터 표현)**이다.
- URL = 자원(명사): `/users`, `/posts/1`
- 행동 = HTTP 메서드(동사): GET / POST / PUT / DELETE

**예시:**
```
❌ 나쁜 설계: GET /getPosts / GET /createPost / GET /deletePost/1
✅ REST식: GET /posts / POST /posts / DELETE /posts/1
```
- URL 보면 **뭘 다루는지**, 메서드 보면 **뭘 하는지** 알 수 있다.

---

#### 1-8. JSON (JavaScript Object Notation)

> 어떤 언어에서든 읽고 쓸 수 있는 공통 데이터 형식.

**사용 이유** \
클라이언트와 서버는 서로 다른 언어와 환경에서 동작할 수 있다.  
JSON은 대부분의 언어에서 쉽게 읽고 쓸 수 있는 표준 포맷이다.

**핵심**
- REST API의 표준 데이터 형식 (XML보다 가볍고 읽기 쉬워서 웹 API 표준이 됨)
- 파이썬 딕셔너리와 비슷하지만 다르다: 키는 반드시 큰따옴표, `True` → `true`, `None` → `null`
- `Content-Type: application/json` 헤더로 데이터 형식 명시

**코드**
```json
{
"name": "kaia",
"age": 25,
"is_student": true
}
``` 

```python
import json

json.dumps({"name": "kaia"}) # 딕셔너리 → JSON 문자열 (직렬화)
json.loads('{"name": "kaia"}') # JSON 문자열 → 딕셔너리 (역직렬화)
```

**흐름**
```text
파이썬 객체
→ JSON 직렬화
→ 네트워크 전송
→ JSON 역직렬화
→ 다시 객체로 사용
```

**비유**
> JSON = 국제 공용어 (한국어 서버, 영어 클라이언트가 둘 다 아는 언어로 대화)

🔗 [파이썬 공식 docs - json 모듈](https://docs.python.org/ko/3/library/json.html)

---

#### 1-9. FastAPI

> 파이썬으로 빠르고 쉽게 API 서버를 만들 수 있는 웹 프레임워크.

**사용 이유** \
HTTP 요청을 직접 소켓 단위로 처리하는 것은 매우 복잡하다.  
FastAPI는 URL과 HTTP 메서드를 함수와 연결해 API 서버를 쉽게 만들 수 있게 해준다.  

**핵심:**
- `@app.get`, `@app.post` 등 **데코레이터**로 HTTP 메서드 + URL 경로 연결
- Pydantic 모델이 요청 Body를 자동으로 검증 → 형식 틀리면 422 반환
- **ASGI 기반** → 비동기(async/await) 지원, 동기 방식 Flask보다 빠름
- **WSGI vs ASGI**: WSGI = 동기 방식(Flask, Django), ASGI = 비동기 지원(FastAPI)
- **Uvicorn**: FastAPI 앱을 실제로 실행시켜주는 ASGI 서버
  
**코드**

```python
from fastapi import FastAPI
from pydantic import BaseModel 

app = FastAPI()

class Item(BaseModel): # 요청 Body의 형태 정의
name: str
price: float

@app.get("/items") # GET /items 요청이 오면 이 함수 실행
def get_items():
return {"items": []}

@app.post("/items") # POST /items 요청이 오면 이 함수 실행
def create_item(item: Item): # Body를 Item 형태로 자동 파싱
return {"created": item}
```

**흐름**
```text
HTTP 요청
→ FastAPI 라우팅
→ 파이썬 함수 실행
→ JSON 응답 반환
```

**비유**
> FastAPI = 가게 운영 시스템 (손님(클라이언트) 주문이 들어오면 어느 직원(함수)한테 갈지 자동으로 연결)

🔗 [FastAPI 공식 문서](https://fastapi.tiangolo.com/ko/)

---

#### 1-10. Pydantic / DTO

> 데이터의 형식과 유효성을 자동으로 검증해주는 라이브러리. 데이터 구조와 타입을 검증하기 위한 모델.

**사용 이유** \
클라이언트가 보내는 데이터가 항상 올바른 형식이라는 보장이 없다. 
Pydantic 모델을 정의해두면 FastAPI가 요청 때 자동 검증할 수 있다.

**핵심**
- `BaseModel`을 상속받아 클래스를 정의 → 타입 검증
- 필수 필드 강제
- 자동완성이 지원
- `.model_dump()` → Pydantic 모델을 딕셔너리로 변환 (직렬화의 한 형태)
- **DTO(Data Transfer Object)**: 계층 간에 데이터를 안전하게 전달하기 위한 객체 <— Pydantic 모델

**코드**

```python
from pydantic import BaseModel

class Post(BaseModel):
title: str # 필수 필드, 문자열이어야 함
content: str
view_count: int = 0 # 선택 필드, 기본값 0

@app.post("/posts")
def create_post(post: Post): # 요청 Body를 Post 형태로 자동 파싱 + 검증
return post
```

**비유**  
> 딕셔너리 = 아무 내용이나 들어갈 수 있는 가방  
> Pydantic 모델 = "이 칸에는 문자열, 저 칸에는 숫자만"이 표시된 정리함  

🔗 [Pydantic 공식 문서](https://docs.pydantic.dev)
🔗 [FastAPI 공식 - Pydantic 모델](https://fastapi.tiangolo.com/ko/tutorial/body/)

---

### Day 2 (05/19) — 로컬 LLM, httpx, 예외처리

#### 2-1. Ollama

> 로컬 환경에서 LLM을 실행할 수 있게 해주는 오픈소스 플랫폼.  
  
**사용 이유** \
클라우드 API 없이 직접 LLM을 실행하기 위해 사용한다.
- 인터넷 없이 사용 가능
- API 비용 없음
- 데이터 외부 유출 최소화

**핵심**
- 모델 다운로드 및 실행 지원
- OpenAI API 호환 엔드포인트 제공
- 로컬 GPU/CPU에서 실행 가능
  
**비유**  
> 클라우드 LLM = 식당에서 시켜먹기, Ollama = 집에서 직접 요리하기  
> 더 느릴 수 있지만, 내 재료, 내 조리법, 외부 공개 없음  

🔗 [엣지 AI란? — Superb AI Blog](https://blog-ko.superb-ai.com/real-time-ai-inference-edge-ai-innovation/)

---

#### 2-2. httpx
  
> Python에서 HTTP 요청을 보내기 위한 클라이언트 라이브러리.

**사용 이유** 
FastAPI 서버가 다른 서버(e.g. Ollama)에 다시 HTTP 요청을 보내야 하기 때문이다.

**핵심** \
- `requests`는 동기만 지원 → FastAPI의 async 환경에서는 `httpx`를 써야 한다
- `httpx.AsyncClient`를 사용하면 비동기로 다른 서버에 요청을 보낼 수 있다

**흐름**
```text
브라우저 → FastAPI
FastAPI → httpx → Ollama
```
- 클라이언트(브라우저, 앱) → FastAPI 서버 → Ollama LLM 서버: FastAPI가 중간에서 다시 HTTP 요청을 보내는 구조
- 브라우저 입장에서는 FastAPI가 **서버**, Ollama 입장에서는 FastAPI가 **클라이언트**
- Ollama한테 요청을 보내는 역할이 `httpx`

**코드:**

```python
import httpx

# 동기 방식
response = httpx.get("http://localhost:11434/v1/models")
print(response.json())

# 비동기 방식 (FastAPI에서 주로 이렇게 씀)
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:11434/v1/chat/completions",
        json={"model": "llama3", "messages": [...]}
    )
```

---

#### 2-3. Payload & 직렬화

> **Payload(페이로드)** 는 HTTP 요청/응답의 본문에 담기는 실제 데이터. 헤더나 메타데이터를 제외한 순수 데이터,  
> **직렬화(Serialization)** 는 메모리 안 객체를 네트워크로 보낼 수 있는 형태(JSON 문자열/bytes)로 변환하는 과정  
> **역직렬화(Deserialization)** 는 받은 JSON/bytes를 다시 파이썬 객체로 복원하는 과정  

**사용 이유** \
네트워크에서는 결국 bytes 형태만 전달할 수 있기 때문이다.  

**핵심**
- `json.dumps()` = 직렬화 (파이썬 → JSON 문자열)
- `json.loads()` = 역직렬화 (JSON 문자열 → 파이썬)
- httpx에서 `json=` 파라미터를 쓰면 자동 직렬화 수행

**코드**

```python
import json

# 직렬화: 파이썬 → JSON 문자열
data = {"model": "llama3", "messages": [{"role": "user", "content": "안녕"}]}
json_str = json.dumps(data)

# 역직렬화: JSON 문자열 → 파이썬
parsed = json.loads(json_str)
  
# httpx는 자동 직렬화
async with httpx.AsyncClient() as client:
    response = await client.post(url, json=data) # json= 쓰면 자동 직렬화
    result = response.json() # 자동 역직렬화
```

**비유**
> 직렬화 = 택배 포장 (물건을 박스에 담기)  
> 역직렬화 = 택배 개봉 (박스에서 물건 꺼내기)  
> 페이로드 = 박스 안에 든 실제 물건  

---

#### 2-4. 스트리밍 (Streaming)

> 응답을 한 번에 모두 보내지 않고, 생성되는 즉시 조금씩 전달하는 방식.

**사용 이유** \
LLM 응답은 토큰 단위로 생성되기 때문에, 스트리밍을 사용하면 사용자는 응답이 생성되는 과정을 실시간으로 볼 수 있다.

**핵심**
- 응답을 청크(chunk) 단위로 받아 처리
- 서버는 SSE: Server-Sent Events 기반으로 데이터를 흘려보냄
- FastAPI에서는 `StreamingResponse`로 응답을 클라이언트에게 스트리밍할 수 있음
- 데이터가 다 왔을 때 `[DONE]` 신호가 옴
  
**코드**

```python
# httpx로 스트리밍 요청
async with httpx.AsyncClient() as client:
    async with client.stream("POST", url, json=payload) as response:
        async for chunk in response.aiter_text():
            if chunk:
                print(chunk, end="", flush=True)

  
# FastAPI에서 스트리밍 응답 반환
from fastapi.responses import StreamingResponse

async def generate():
    async with httpx.AsyncClient() as client:
        async with client.stream(...) as resp:
            async for chunk in resp.aiter_text():
                yield chunk
  
@app.post("/chat")
async def chat():
return StreamingResponse(generate(), media_type="text/event-stream")
```

**비유**
> 일반 응답 = 주방에서 밥 다 차려진 다음에 한꺼번에 서빙  
> 스트리밍 = 요리사가 만들면서 바로바로 접시에 올려 내보냄  

---

#### 2-5. 컨텍스트 매니저 (`with`문)
  
> 자원 관리 객체. 파일, 네트워크 연결 등 리소스를 쓸 때 코드 블록이 끝나면 자동으로 닫아주는 것
  
**사용 이유** \
자원의 생명 주기를 코드 구조로 강제하기 위함. 
직접 관리하게 되면 실수나 예외 상황에서 닫는 코드가 누락될 수 있는데, 자원 누수나 정리 누락 같은 실수를 방지할 수 있음.

**핵심**
- `with`가 끝나면 `__exit__`이 자동 호출됨 → 예외가 나도 반드시 닫힘
- `httpx.AsyncClient()`는 특히 커넥션 풀을 관리하므로, 반드시 `async with`로 써야 함
- `with`를 안 쓰면: 파일/연결이 안 닫혀 리소스 낭비

**코드**

```python
# ✅ with 사용 — 자동으로 닫힘
with open("data.txt", "r") as file:
content = file.read()
# with 블록 벗어나는 순간 file.close() 자동 호출

# httpx 비동기 예시
async with httpx.AsyncClient() as client:
response = await client.get("http://localhost:11434/...")
# 여기서 client가 자동으로 정리됨
```

```python
# with문의 동작 원리
class ResourceContext:
    def __enter__(self):
        print("시작 준비")
        return "실제 코드 실행"

def __exit__(self, exception_type, exception_value, traceback):
    print("마무리 정리") # 예외가 나도 항상 여기까지 옴
    
    with ResourceContext() as value:
        print(value)
```

🔗 [Python contextlib 공식 문서](https://docs.python.org/ko/3/library/contextlib.html)

---

#### 2-6. 예외 처리

> 프로그램 실행 중 발생 가능한 오류 상황에 대응하는 방법

**사용 이유** \
프로그램에서 예상치 못한 상황에 대비해 안정성을 높이고 오류를 관리하기 위해서. 
강제 종료될 만한 상황이 있을 때 강제 종료 상황을 만들지 않게 하기 위해 사용.
  
**early return (조기 반환)**

- 반환을 조기에 진행해서, 뒷코드 구조를 단순하게 만들어줌
- 코드 실행을 조건에 따라 빠르게 중단시켜 복잡한 조건문이나 중첩된 if문을 피하면서 **가독성을 높임**
- 언제? 일반적인 조건 분기, 유효성 검사

```python
# early return 미적용 — 끝까지 내려가는 구조
def check_age(age):
    if age >= 18:
      result = '성인입니다.'
    else:
        result = '미성년자입니다.'
    return result
  
# ✅ early return 적용 — 엣지케이스 먼저 처리
def check_age(age):
    if age < 18:
        return '미성년자입니다.'
        return '성인입니다.'
```

**try-except**
- try 블록 내에서 코드가 동작하는 동안 발생할 수 있는 **예외를 처리**
- 조기 리턴은 코드를 명확하고 간결하게 하기 위함(가독성), try-except는 에러가 발생했을 때를 처리하기 위함
- 언제? 네트워크 요청, 파일 I/O, 외부 API 호출

```python
# 기본 구조
try:
    # 동작 코드
    pass
    except Exception as error:
    return str(error)

# 실전: LLM 네트워크 요청
try:
    response = await client.post(url, json=payload, timeout=30.0)
    result = response.json()
    except httpx.TimeoutException:
        print("요청 시간 초과")
    except httpx.ConnectError:
        print("Ollama 서버에 연결할 수 없음. ollama serve가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"알 수 없는 오류: {e}")
finally:
    print("요청 완료") # 성공/실패 무관하게 항상 실행
```

**비유**  
> try = 시도해보기  
> except = 실패하면 이렇게 대응하기  
> finally = 성공이든 실패든 꼭 해야 할 마무리  

🔗 [Python 예외 처리 공식 문서](https://docs.python.org/ko/3/tutorial/errors.html)

---

### Day 3 (05/20) — 데이터베이스 기초

#### 3-1. Database / DBMS

> DB = 구조화된 데이터의 모음(개념), DBMS = 그 DB를 실제로 관리하는 시스템(구현체).

**사용 이유** \
데이터를 단순히 저장하는 게 아니라, 일정한 구조로 정리하고 저장·조회·수정·삭제가 가능한 형태로 관리하기 위해.

**핵심**
- DB ≠ DBMS
  - DB는 개념, DBMS는 실제 소프트웨어 (MySQL, PostgreSQL, Oracle 등)
- DBMS의 역할: 저장 / 조회 / 수정 / 삭제 / 동시성 관리 / 트랜잭션 관리

**비유**
> DB = 잘 정리된 서류함(개념)
> DBMS = 서류함을 관리해주는 직원(시스템)

---

#### 3-2. 정형 데이터 vs 비정형 데이터

> 정형 = 행/열 구조, schema 있음, SQL 사용 가능.
> 비정형 = 구조 없음, 텍스트·이미지·음성·영상.

**사용 이유** \
어떤 데이터를 다루느냐에 따라 어떤 DB를 쓸지가 결정된다.

**핵심**
- 정형(Structured): id / name / age 같은 표 형태, **schema 존재**
- 비정형(Unstructured): 이미지, 영상, 음성, 자유 텍스트 → RDB로 다루기 어렵다
- AI 관점: 모델 학습 데이터는 주로 비정형 → 전처리 과정에서 정형화

**비유**
> 정형 = 엑셀 표
> 비정형 = 카카오톡 대화 내용

---

#### 3-3. RDB (Relational Database, 관계형 데이터베이스)

> 데이터를 테이블(표) 형태로 관리하는 DB.

**사용 이유** \
데이터를 표 형태로 구조화하여 데이터 간 관계를 명확히 하고 효율적으로 다루기 위해.

**핵심**
- 구성: 테이블(table) + 행(row) + 열(column)
- 강사 강조: "관계형이라는 말보다 표 기반이라고 이해하는 게 더 직관적"
- RDB의 대표: MySQL, PostgreSQL, SQLite, Oracle

**비유**
> RDB = 여러 개의 연결된 엑셀 시트

---

#### 3-4. SQL

> DB를 다루기 위한 질의 언어. 데이터를 생성, 조회, 수정, 삭제하는 언어.

**사용 이유** \
RDB의 데이터를 조회·추가·수정·삭제하기 위해.

**핵심**

| 종류 | 이름 | 역할 | 주요 명령 |
|------|------|------|-----------|
| DDL | 데이터 정의어 | 구조 생성, 변경 | CREATE, ALTER, DROP |
| DML | 데이터 조작어 | 데이터 조작 | SELECT, INSERT, UPDATE, DELETE |
| DCL | 데이터 제어어 | 권한 부여 | GRANT, REVOKE |

**코드**
```sql
-- DDL: 테이블 생성
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER
);

-- DML: 데이터 조작
SELECT * FROM users WHERE age > 20;
INSERT INTO users (name, age) VALUES ('다은', 25);
```

---

#### 3-5. 정규화 (Normalization) & 이상현상 (Anomaly)

> 정규화 = 이상현상을 줄이기 위해 데이터를 올바르게 구조화하는 과정.

**사용 이유** \
DB 구조가 잘못되면 이상현상(데이터 오류)이 발생한다. 희귀하지만 발생하면 시스템 전체에 치명적.

**핵심**
- 이상현상(Anomaly) 종류: 수정 이상 / 삽입 이상 / 삭제 이상
- anomaly detection, 예외처리와 같은 맥락 — 자주 발생하지 않지만 한 번 발생하면 치명적

유사 용어 혼동 주의:

| 용어 | 분야 | 뜻 |
|------|------|-----|
| Normalization | DB | 데이터 정규화 |
| Regularization | ML/딥러닝 | 과적합 방지 규제 |
| Canonicalization | 일반 | 표현을 표준 형태로 변환 |

---

#### 3-6. ERD (Entity Relationship Diagram)

> DB 설계도. 어떤 테이블이 있고 어떻게 연결되는지를 시각화한 것.

**사용 이유** \
프로젝트 시작 전 "어떤 데이터를 어떤 구조로 어떻게 연결할 것인가"를 설계하는 단계. 개발자 간 공통 언어.

**핵심**
- ERD는 반드시 볼 줄 알아야 한다 — 개발자 간 통용되는 설계도
- 프로젝트 시작 전 ERD 그리기 → 구현 중 방황 방지
- 설계를 명확히 하고, 소통을 도우며 설계 의도를 공유하고 유지

**종류**

**IE(Information Engineering) / Crow's Foot Notation** — 까마귀 발 표기법

엔티티 간 관계:
- `1:1` A 1개 ↔ B 1개. '특성'
- `1:N` A 1개 ↔ B 여러 개. '소속', '귀속'
- `N:M` A 여러 개 ↔ B 여러 개

기호표:

| 기호 | 의미 |
|------|------|
| □ 사각형 | 엔티티 |
| ○ 타원 | 0개 |
| \| 해쉬 마크 | 1개 |
| < 까마귀 발 | 2개 이상 (n) |
| — 실선 | Identifying: 부모 테이블의 기본 키를 자식 테이블의 기본 키로 사용. A가 없으면 B가 존재할 수 없는 관계 |
| - - - 점선 | Non-Identifying: A가 없어도 B가 존재할 수 있는 관계 |

**ERD 작성 순서**
1. 엔터티를 그린다
2. 엔터티를 적절하게 배치한다
3. 엔터티 간 관계를 설정한다 (식별자 관계, 중복/Circle 관계 금지)
4. 관계명을 기술한다 (현재형, 포괄적 용어 지양)
5. 관계의 참여도 및 필수여부를 기술한다

**비유**
> ERD = 건물 짓기 전 설계 도면.
> 도면 없이 짓기 시작하면 나중에 뜯어고쳐야 한다.

🔗 [데이터 모델링 - IE/Crow's Foot 표기법](https://ppomelo.tistory.com/51)

---

#### 3-7. Index

> 조회 성능을 최적화하는 "목차". Index 없으면 Full Scan (처음부터 끝까지 전부 탐색).

**사용 이유** \
데이터가 많아질수록 조회가 느려진다. Index는 특정 컬럼에 대한 검색 속도를 빠르게 해준다.

**핵심**
- Index 없을 때: Full Scan → 전체 데이터를 처음부터 탐색 (느림)
- Index 있을 때: B-Tree 등의 자료구조로 빠르게 찾음
- 트레이드오프: Index는 저장 공간을 따로 차지하고, 쓰기(INSERT/UPDATE/DELETE) 시 인덱스도 업데이트해야 함 → 모든 컬럼에 다 걸면 쓰기 성능 저하. **자주 조회하는 컬럼에만** 거는 이유.

**Index 종류**
- 클러스터형: 데이터가 인덱스 순서대로 물리적으로 정렬되어 저장
- 비클러스터형: 인덱스와 데이터 물리적으로 분리. 인덱스 테이블에 포인터만 저장 → 원본 데이터 참조

**배경: Index를 가능하게 하는 자료구조**

> Index 내부 동작을 이해하려면 Tree 구조를 알아야 한다.

- **BST (Binary Search Tree)**: 노드의 왼쪽 < 현재값 < 오른쪽. 문제점: 편향 트리 발생 시 성능 저하
- **B-Tree**: DB와 파일 시스템 전용. 하나의 노드가 여러 키와 자식을 가질 수 있고, 리프 노드가 모두 같은 깊이 → 균형 유지. 검색·삽입·삭제 모두 O(log n)
- **B+Tree**: MySQL InnoDB에서 인덱스 관리에 사용. B-Tree와 차이: 모든 레코드가 리프 노드에만 저장, 내부 노드에는 탐색 경로(키)만 존재 → 범위 검색에 유리

**비유**
> 책갈피. "종이 사전에서 단어 찾기" → Index 없으면 처음부터 끝까지 읽어야 함.

---

#### 3-8. Transaction

> 전부 성공하거나 전부 실패해야 하는 DB 작업의 단위. 데이터 무결성 보장.

**사용 이유** \
계좌 이체처럼 중간에 실패하면 안 되는 작업이 있다.
Transaction이 없으면 A 계좌에서 돈이 빠졌는데 B 계좌에 안 들어오는 상황이 발생할 수 있다.

**핵심**
- ACID 속성: Atomicity(원자성) / Consistency(일관성) / Isolation(격리성) / Durability(지속성)
- 핵심: 중간 상태가 없다. 커밋(commit) 아니면 롤백(rollback).
- 트랜잭션은 필요한 최소 범위에서 사용하는 것이 좋음
- DBMS에서는 모든 쿼리가 트랜잭션 컨텍스트 안에서 실행된다

**트랜잭션 상태**

![[Pasted image 20260528233830.png]]

- **Active**: 실행 중 상태. 아직 성공/실패가 결정되지 않음
- **Partially Committed**: commit 명령이 도착한 상태. 논리적으로 SQL 명령 성공했지만 디스크에 영구 기록 전 → 장애 발생 시 완료되지 않을 수 있음
- **Committed**: 트랜잭션이 완전히 성공한 상태. 되돌릴 수 없다
- **Failed**: 정상적으로 더이상 진행될 수 없는 상태
- **Aborted**: 트랜잭션 취소되어 모든 변경 사항이 롤백됨

**비유**
> 계좌 이체: A -5000, B +5000 → 이 두 개가 하나의 Transaction. 하나만 성공하면 안 됨.

---

#### 3-9. NoSQL

> RDB(관계형 구조)를 따르지 않는 DB. 유연한 구조, 빠른 처리, 비정형 데이터 친화적.

**사용 이유** \
정형 데이터가 아닌 경우, 혹은 대용량 데이터를 빠르게 처리해야 할 때 RDB보다 적합할 수 있다.

**핵심**
- NoSQL = "Not Only SQL" — "SQL이 없다"는 뜻이 아니라 **관계형 구조(테이블/스키마)를 따르지 않는다**는 뜻
- MongoDB도 자체 쿼리 언어(MQL)가 있어 SQL처럼 조회 가능 → NoSQL인 이유는 관계형 구조가 아니어서
- 대표 예시: MongoDB(문서), Redis(키-값), Cassandra(컬럼), Neo4j(그래프)
- 유연한 schema → 구조가 자주 바뀌는 데이터에 유리

**비유**
> RDB = 칸이 정해진 표
> NoSQL = 형식이 자유로운 메모장

---

### Day 4 (05/21) — 구조 개선, 프론트엔드

#### 4-1. 구조 개선 — 왜 선택의 이유가 중요한가

> 소프트웨어 개발자의 업무 = 선택과 그에 따른 책임. 선택에는 항상 이유가 있어야 한다.

**사용 이유** \
구조를 바꾸는 것도 하나의 선택이다. 이유 없이 바꾸면, 그 선택 이후에 투자되는 시간과 노력을 정당화할 수 없다.

**핵심**
- "왜 이렇게 했어요?" → 이유 없음 = 선택에 고민이 없었다는 뜻
- 업무 = 선택 + 그에 따른 책임 → 개발자는 선택 전문가다
- main.py 하나로 불편함·어려움·불가능이 없다면 구조를 바꿀 이유가 없다 — 이유가 생길 때 바꾸는 것
- 스스로 깨닫고 찾고 느끼는 경험이 있느냐 없느냐가 중요
- 남의 경험을 사게 되는 것 = 배운다는 것. 지금은 수출 < 수입(배우는 게 많음) 단계지만, 경험이 쌓일수록 수출 > 수입으로 바뀐다

**비유**
> 선택 = 뽑기가 아니라 근거 있는 판단. 10년, 20년 선택 훈련을 하면 다른 분야에서도 잘한다.

---

#### 4-2. 디자인 패턴 — Route · Controller · Model

> 자주 사용하는 소프트웨어 설계 형태를 정형화해 유형별로 만들어 둔 템플릿.

**사용 이유** \
복잡도를 낮추고 확장성·재사용성·디버깅 편의를 높이기 위해서다. 면접 단골 질문이기도 하다.

**핵심**
- 코드에 구조적인 문제가 있다는 건 소프트웨어의 복잡도(Complexity)가 올라갔다는 신호
- 리팩토링 = 기능의 변화 없이 구조만 변화 (기능 추가 ≠ 리팩토링)
- Router = 어떤 요청이 오면 어떤 컨트롤러로 넘길지 결정 / Controller = 비즈니스 로직 처리 / Model = 데이터(DB) 관리

| 레이어 | 역할 | 비유 |
|--------|------|------|
| Router | 경로 분배 | 안내 데스크 |
| Controller | 비즈니스 로직 | 실무 담당자 |
| Model | 데이터 관리 | 자료 창고 담당자 |

**코드**
```
프로젝트 구조:
├── main.py         ← 앱 시작점, 라우터 등록
├── routers/
│   └── user.py     ← 요청 경로 정의, Controller로 위임
├── controllers/
│   └── user.py     ← 비즈니스 로직 처리
└── models/
    └── user.py     ← DB 관련 로직
```

🔗 [FastAPI 공식 - Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)

---

#### 4-3. 미들웨어 (Middleware)

> 두 소프트웨어 층 사이에 있는 층. 모든 요청에 공통으로 끼어드는 장치.

**사용 이유**  \
특정 경로가 아니라 모든 요청에 공통으로 처리할 것들(로깅, 인증, 실행 시간 측정, CORS 등)을 한 곳에서 처리하기 위해서다.

**핵심**
- 미들웨어는 맥락에 따라 가리키는 대상이 다르다 — 항상 "어느 두 층 사이"인지 파악해야 함
- OS·웹앱 맥락: Web App ↔ Middleware(FastAPI, Uvicorn) ↔ OS
- ASGI·핸들러 맥락(FastAPI에서 주로 말하는 것): request handler ↔ Middleware(CORS 등) ↔ ASGI server(Uvicorn)

**코드**
```python
from fastapi import FastAPI
import time

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

#### 4-4. HTML

> 마크업 언어 — 태그로 문서 구조를 표현. 브라우저가 파싱 후 렌더링한다.

**사용 이유** \
웹 브라우저에서 보이는 화면의 뼈대이자, 내용을 담는 구조체이기 때문이다.

**핵심**
- 브라우저의 두 가지 역할: Parser(이해, HTML을 읽어 구조 파악) / Renderer(출력, 파싱한 구조를 화면에 그림)
- HTML을 소비(보는) 방법: 직접 파일 열기(`open index.html`) / 서버에 접속하기(`python -m http.server`, FastAPI Static Files)

🔗 [MDN - HTML](https://developer.mozilla.org/ko/docs/Web/HTML)

---

#### 4-5. CSS

> HTML 문서에 시각적 스타일을 입히는 언어. 뼈대(HTML) 위에 옷을 입히는 것.

**사용 이유** \
HTML만으로는 정보는 담기지만 시각적으로 매력적이지 않다. CSS로 레이아웃·색상·폰트 등을 제어한다.

**핵심**
- CSS를 잘하는 인간 전문가가 드물고, 왜 이렇게 보이는지 원인을 찾는 디버깅도 어렵다
- 요즘은 AI가 CSS 구현을 인간 전문가 수준 이상으로 처리 — 미감(무엇을 만들지)은 내가 정하고 구현은 AI를 활용

🔗 [MDN - CSS](https://developer.mozilla.org/ko/docs/Web/CSS)

---

#### 4-6. JavaScript & Fetch

> 브라우저에서 실행되는 언어. 동적 콘텐츠 구현 + 서버와 통신(Fetch).

**사용 이유** \
HTML+CSS는 정적이다. JS가 있어야 버튼 클릭, 데이터 로딩, 화면 업데이트 같은 동작이 가능하다.

**핵심**
- 탄생 배경: "브라우저에서 코드를 실행하고 싶다"는 요구에서 시작된 동적 웹의 출발점
- Fetch = 브라우저에서 서버로 HTTP 요청을 보내는 JS 함수

🔗 [Fetch 정리](https://www.notion.so/adapterz/Fetch-2df394a480618063a7f4e57f43082e61)

---

#### 4-7. CORS (Cross-Origin Resource Sharing)

> 브라우저가 다른 출처의 서버에 요청할 때 적용하는 보안 정책. 서버가 허용해야 풀린다.

**사용 이유** \
JS(Fetch)가 자유롭게 모든 서버에 요청할 수 있으면 개인정보가 다른 서버로 유출될 수 있다. 브라우저가 이를 막아준다.

**핵심**
- 접속을 통제하는 주체는 브라우저 정책 (서버가 아님!) — 프로토콜+호스트+포트가 모두 같아야 동일 출처
- 풀어주는 주체는 서버 정책 — 응답 헤더에 허용 출처를 명시(`Access-Control-Allow-Origin`)
- 실무 대화 예시: "서버 접속이 안 돼요." → "CORS 뚫어주세요."
- Preflight 요청: POST 등 데이터 변경 가능 메서드는 실제 요청 전 OPTIONS로 먼저 확인

**코드**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

🔗 [CORS 정리](https://www.notion.so/adapterz/CORS-196394a48061808f8e4fcbb2b3ec076a) · [MDN - CORS](https://developer.mozilla.org/ko/docs/Web/HTTP/CORS)

---

#### 4-8. Streamlit

> 파이썬 코드로 프론트엔드를 만드는 라이브러리. HTML/CSS/JS 없이 웹 UI 구현 가능.

**사용 이유** \
AI 엔지니어가 프론트 언어를 깊게 배우지 않고도 결과물을 시각화하고 인터랙티브 UI를 만들 수 있다.

**핵심**
- 파이썬 코드만으로 버튼, 입력창, 차트, 텍스트 출력 가능
- 위클리 챌린지 선택 항목: FastAPI 백엔드 + Streamlit 프론트엔드 조합 가능
- AI + 바이브코딩으로 빠르게 프로토타이핑 가능

🔗 [Streamlit 공식 문서](https://streamlit.io)

---

## 🔗 이번 주 개념 흐름

```
[브라우저 / 클라이언트]
        ↓  HTTP 요청 (메서드 + URL + Body)
[FastAPI 서버]  ← Pydantic으로 요청 검증
        ↓  httpx로 HTTP 요청 (직렬화)
[Ollama LLM 서버]
        ↓  스트리밍 응답 (token by token)
[FastAPI]  ← 예외처리 / with문으로 리소스 관리
        ↓  StreamingResponse
[브라우저]

        + 중간에 DB 저장 (Day3)
        + 서비스 구조화 / 미들웨어 (Day4)
        + TLS 암호화 → HTTPS (Day5)
```

이번 주에는 단순히 개별 기술을 배우는 것이 아니라,
브라우저의 요청이 실제 AI 응답이 되어 돌아오기까지의 흐름을
하나의 시스템 관점에서 연결해보는 데 집중했다.

HTTP와 FastAPI로 API 서버를 만들고,
httpx를 통해 로컬 LLM 서버(Ollama)와 통신했으며,
스트리밍 응답과 예외처리를 통해 실제 서비스 구조를 경험했다.

여기에 DB로 데이터를 연결하고, 디자인 패턴으로 구조를 정리하고,
HTTPS로 전체 통신을 암호화하면 백엔드의 최소 가동 구조가 완성된다.

---
