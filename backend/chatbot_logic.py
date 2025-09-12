# chatbot_logic.py (수정 완료)

import os
import json
import re
from typing import TypedDict, List, Optional, Dict
import sys
import traceback

# Pydantic 및 LangChain 호환성을 위한 임포트
from pydantic import BaseModel, Field, PrivateAttr

# --- 변경된 부분: 라이브러리 임포트 수정 ---
import openai # openai 라이브러리 직접 임포트
from langchain_openai import OpenAIEmbeddings # 임베딩 모델은 LangChain 호환성을 위해 유지
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.callbacks.base import Callbacks
from langchain_core.runnables import RunnableLambda # 커스텀 함수를 체인에 연결하기 위해 추가

# FlashRank 임포트
try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    print("FlashRank 또는 관련 모듈을 찾을 수 없습니다. Reranking 기능이 비활성화됩니다.")
    Ranker = None

# --- 설정 및 모델 정의 ---
MAX_CORRECTION_ATTEMPTS = 3

# --- 전역 변수 및 헬퍼 함수 ---
retrievers = {}
approved_templates = []
rejected_templates = []


# --- 추가된 부분: OpenAI API 직접 호출 헬퍼 함수 ---
# 참고: 'gpt-5', 'gpt-4.1'은 실제 OpenAI 모델명이 아닙니다.
# 실제 사용 시 'gpt-4o', 'gpt-4-turbo' 등 사용 가능한 모델명으로 변경해야 합니다.

def call_openai_api(prompt_value, model: str, temperature: float = 0.0) -> str:
    """일반적인 텍스트 응답을 위한 OpenAI API 호출"""
    try:
        # [수정] prompt_value를 API가 요구하는 메시지 형식으로 변환
        messages = prompt_value.to_messages()
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API for model {model}: {e}")
        return f"API 호출 중 오류 발생: {e}"

def call_openai_api_for_json(prompt_value, model: str) -> str:
    """JSON 응답을 위한 OpenAI API 호출 (JSON 모드 사용)"""
    try:
        # [수정] prompt_value를 API가 요구하는 메시지 형식으로 변환
        messages = prompt_value.to_messages()
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API for JSON with model {model}: {e}")
        # [수정] KeyError 방지를 위해 status를 포함한 오류 JSON 반환
        return json.dumps({"status": "error", "reason": f"API 호출 실패: {e}"})


class CustomRuleLoader(BaseLoader):
    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        self.file_path = file_path
        self.encoding = encoding
    
    def load(self) -> List[Document]:
        docs = []
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
        except FileNotFoundError:
            print(f"🚨 경고: '{self.file_path}' 파일을 찾을 수 없습니다.")
            return []
        
        rule_blocks = re.findall(r'\[규칙 시작\](.*?)\[규칙 끝\]', content, re.DOTALL)
        for block in rule_blocks:
            lines = block.strip().split('\n')
            metadata = {}
            page_content = ""
            is_content_section = False
            
            for line in lines:
                if line.lower().startswith('content:'):
                    is_content_section = True
                    page_content += line[len('content:'):].strip() + "\n"
                    continue
                if is_content_section:
                    page_content += line.strip() + "\n"
                else:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
            
            if page_content:
                docs.append(Document(page_content=page_content.strip(), metadata=metadata))
        
        return docs

class TemplateAnalysisResult(BaseModel):
    status: str = Field(description="템플릿의 최종 상태")
    reason: str = Field(description="상세한 판단 이유")
    evidence: Optional[str] = Field(None, description="판단 근거 규칙들의 rule_id")
    suggestion: Optional[str] = Field(None, description="개선 제안")

class FlashRankRerank(BaseDocumentCompressor):
    _ranker: Ranker = PrivateAttr()
    top_n: int = 3
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if Ranker:
            self._ranker = Ranker()
    
    class Config:
        arbitrary_types_allowed = True
    
    def compress_documents(self, documents: List[Document], query: str, callbacks: Callbacks | None = None) -> List[Document]:
        if not documents or not Ranker:
            return documents[:self.top_n]
        
        rerank_request = RerankRequest(
            query=query,
            passages=[{"id": i, "text": doc.page_content, "metadata": doc.metadata} for i, doc in enumerate(documents)]
        )
        reranked_results = self._ranker.rerank(rerank_request)
        
        final_docs = []
        for item in reranked_results[:self.top_n]:
            doc = documents[item['id']]
            doc.metadata["relevance_score"] = item['score']
            final_docs.append(doc)
        
        return final_docs

class Variable(BaseModel):
    name: str = Field(description="추출된 변수의 한글 이름 (예: 매장명, 폐점일자). `#{{}}`에 들어갈 부분입니다.")
    original_value: str = Field(description="원본 텍스트에서 추출된 실제 값")
    description: str = Field(description="해당 변수에 대한 간단한 한글 설명 (사용자가 이해하기 쉽도록)")

class ParameterizedResult(BaseModel):
    parameterized_template: str = Field(description="특정 정보가 #{{변수명}}으로 대체된 최종 템플릿")
    variables: List[Variable] = Field(description="추출된 변수들의 목록")

class StructuredTemplate(BaseModel):
    title: str = Field(description="템플릿의 제목 또는 첫 문장")
    body: str = Field(description="제목과 버튼 텍스트를 제외한 템플릿의 핵심 본문 내용. 줄바꿈이 있다면 \\n으로 유지해주세요.")
    buttons: Optional[List[tuple[str, str]]] = Field(None, description="템플릿에 포함될 버튼 리스트. 예: [('웹사이트', '자세히 보기')]")


def load_line_by_line(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            items = [line.strip() for line in f if line.strip()]
        return items
    except FileNotFoundError:
        return []

def load_by_separator(file_path: str, separator: str = '---') -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        items = [section.strip() for section in content.split(separator) if section.strip()]
        return items
    except FileNotFoundError:
        return []


def structure_template_with_llm(template_string: str) -> StructuredTemplate:
    parser = JsonOutputParser(pydantic_object=StructuredTemplate)

    system_prompt = '''당신은 주어진 텍스트를 분석하여 핵심 구성 요소로 구조화하고, 본문을 사용자가 읽기 쉽게 편집하는 전문가입니다.

# 지시사항:
1.  텍스트의 핵심 의도를 'title'로 추출합니다. 길이는 짧고 간결하게.
2.  텍스트에 버튼 정보가 있다면 분석하여 'buttons' 리스트를 생성합니다. 없다면 빈 리스트 `[]`를 반환합니다.
3.  버튼은 ['버튼종류', '버튼이름'] 형식의 튜플이어야 합니다. 버튼 종류는 내용에 맞게 '웹사이트', '앱링크', '전화하기' 등으로 추론하세요.
4.  제목과 버튼을 제외한 나머지 내용을 'body'의 기본 재료로 사용합니다.
5.  'body'의 내용을 사용자가 이해하기 쉽게 **재구성하고 가독성을 높여주세요.** 다음 규칙을 따르세요:
    -   **문맥의 흐름을 파악하여 의미 단위로 문단을 나누고, 줄바꿈(`\n`)을 추가하세요.**
    -   나열되는 항목(예: `▶`, `※`)이 있다면 글머리 기호('-')를 사용하여 목록으로 만드세요.
    -   전체적으로 문장을 간결하고 명확하게 다듬어주세요.
6.  최종 결과는 반드시 지정된 JSON 형식으로만 출력해야 합니다. 서론이나 추가 설명은 절대 포함하지 마세요.

# 학습 예시 (Few-shot Examples)
---
## 예시 1
-   **원본 텍스트:** "안녕하세요, #{{수신자}}님! 회원이 되신 것을 환영합니다. 🔔 신규 회원 안내 사항 🔔 #{{안내사항}}"
-   **올바른 출력 (JSON):**
    ```json
    {{
        "title": "신규 회원 가입 환영",
        "body": "안녕하세요, #{{수신자}}님!\\n\\n회원이 되신 것을 환영합니다.\\n\\n🔔 신규 회원 안내 사항 🔔\\n#{{안내사항}}",
        "buttons": []
    }}
    ```
---
## 예시 2
-   **원본 텍스트:** "안녕하세요. #{{수신자명}}님, 멤버십에 가입 완료되었습니다. ▶ 멤버십 : #{{멤버십}} ▶ 가입일 : #{{가입일}} ▶ 멤버십 혜택 안내 : #{{멤버십 혜택 안내}} 언제든 궁금한 사항이 있다면 홈페이지에 문의해주세요. 감사합니다."
-   **올바른 출력 (JSON):**
    ```json
    {{
        "title": "멤버십 가입 완료 안내",
        "body": "안녕하세요, #{{수신자명}}님.\\n멤버십 가입이 완료되었습니다.\\n\\n- 멤버십: #{{멤버십}}\\n- 가입일: #{{가입일}}\\n- 멤버십 혜택: #{{멤버십 혜택 안내}}\\n\\n언제든 궁금한 사항이 있다면 홈페이지에 문의해주세요.\\n\\n감사합니다.",
        "buttons": []
    }}
    ```
---
## 예시 3
-   **원본 텍스트:** "안녕하세요. #{{수신자명}} 고객님, 회원이 되신 것을 진심으로 환영합니다. 회원가입 혜택으로 지금 #{{상품}} 구매 시 할인된 가격으로 구매하실 수 있습니다. ※ 이 메시지는 고객님의 동의에 의해 전송되는 메시지입니다."
-   **올바른 출력 (JSON):**
    ```json
    {{
        "title": "회원가입 환영 및 혜택 안내",
        "body": "안녕하세요, #{{수신자명}} 고객님.\\n회원이 되신 것을 진심으로 환영합니다.\\n\\n회원가입 혜택으로 지금 #{{상품}} 구매 시 할인된 가격으로 구매하실 수 있습니다.\\n\\n※ 이 메시지는 고객님의 동의에 의해 전송되는 메시지입니다.",
        "buttons": []
    }}
    ```
---'''

    human_prompt = '''# 실제 작업 요청
-   **원본 텍스트:** {raw_text}
-   **출력 형식 (JSON):** {format_instructions}'''

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    chain = prompt | RunnableLambda(lambda p: call_openai_api_for_json(p.to_messages(), model="gpt-4.1")) | parser
    
    try:
        structured_data_dict = chain.invoke({
            "raw_text": template_string,
            "format_instructions": parser.get_format_instructions()
        })
        return StructuredTemplate(**structured_data_dict)
    except Exception as e:
        print(f"Error during structuring template: {e}")
        return StructuredTemplate(
            title=template_string.split('\n')[0].strip(),
            body=template_string,
            buttons=[]
        )

def parameterize_template(template_string: str) -> Dict:
    parser = JsonOutputParser(pydantic_object=ParameterizedResult)
    prompt = ChatPromptTemplate.from_template(
        '''당신은 주어진 텍스트를 재사용 가능한 템플릿으로 변환하는 전문가입니다.
        주어진 텍스트에서 고유명사, 날짜, 장소, 숫자 등 구체적이고 바뀔 수 있는 정보들을 식별하여, 의미 있는 한글 변수명으로 대체해주세요.
        # 지시사항
        1. 텍스트의 핵심 정보(누가, 언제, 어디서, 무엇을, 어떻게 등)를 파악합니다.
        2. 파악된 정보를 `#{{변수명}}` 형태로 대체하여 재사용 가능한 템플릿을 생성합니다. 변수명은 사용자가 이해하기 쉬운 한글로 작성하세요.
        3. 원본 값과 변수명, 그리고 각 변수에 대한 설명을 포함하는 변수 목록을 생성합니다.
        4. 최종 결과를 지정된 JSON 형식으로만 출력해야 합니다. 그 외의 설명은 절대 추가하지 마세요.
        # 원본 텍스트:
        {original_text}
        # 출력 형식 (JSON):
        {format_instructions}
        '''
    )
    chain = prompt | RunnableLambda(lambda p: call_openai_api_for_json(p, model="gpt-4.1")) | parser
    try:
        result = chain.invoke({
            "original_text": template_string,
            "format_instructions": parser.get_format_instructions(),
        })
        if not isinstance(result, dict):
            result = {"parameterized_template": template_string, "variables": []}
        if "parameterized_template" not in result:
            result["parameterized_template"] = template_string
        if "variables" not in result:
            result["variables"] = []
        return result
    except Exception as e:
        print(f"Error during parameterization: {e}")
        return {"parameterized_template": template_string, "variables": []}

def initialize_system():
    global retrievers, approved_templates, rejected_templates
    
    if 'compliance' in retrievers and retrievers['compliance'] is not None:
        print("시스템이 이미 초기화되었습니다.")
        return
        
    print("서버 시작: 시스템 초기화를 진행합니다...")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        openai.api_key = api_key
        
        data_dir = 'data'
        vector_db_path = "vector_db"
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        approved_templates = load_line_by_line(os.path.join(data_dir, "approved_templates.txt"))
        rejected_templates = load_by_separator(os.path.join(data_dir, "rejected_templates.txt"))
        
        docs_compliance = CustomRuleLoader(os.path.join(data_dir, "compliance_rules.txt")).load()
        docs_generation = CustomRuleLoader(os.path.join(data_dir, "generation_rules.txt")).load()
        docs_whitelist = [Document(page_content=t) for t in approved_templates]
        docs_rejected = [Document(page_content=t) for t in rejected_templates]
        
        from chromadb.config import Settings
        client_settings = Settings(anonymized_telemetry=False)
        
        def create_db(name, docs):
            if not docs:
                print(f"🚨 '{name}'에 대한 문서가 없어 DB 생성을 건너뜁니다.")
                return None
            try:
                print(f"🔍 '{vector_db_path}'에서 기존 '{name}' 컬렉션을 불러오거나 새로 생성합니다...")
                db = Chroma(
                    collection_name=name,
                    persist_directory=vector_db_path,
                    embedding_function=embeddings,
                    client_settings=client_settings
                )
                if db._collection.count() == 0:
                     print(f"🤔 '{name}' 컬렉션이 비어있어 문서를 새로 추가합니다.")
                     db.add_documents(docs)
                     db.persist()
                print(f"✅ '{name}' 컬렉션을 성공적으로 준비했습니다. (항목 수: {db._collection.count()})")
                return db
            except Exception as e:
                print(f"⚠️ DB 준비 중 오류 발생({e}). DB를 새로 생성합니다.")
                db = Chroma.from_documents(
                    docs, 
                    embeddings, 
                    collection_name=name, 
                    persist_directory=vector_db_path, 
                    client_settings=client_settings
                )
                db.persist() 
                print(f"💾 '{name}' 컬렉션이 '{vector_db_path}'에 저장되었습니다.")
                return db

        db_compliance = create_db("compliance_rules", docs_compliance)
        db_generation = create_db("generation_rules", docs_generation)
        db_whitelist = create_db("whitelist_templates", docs_whitelist)
        db_rejected = create_db("rejected_templates", docs_rejected)
        
        def create_hybrid_retriever(vectorstore, docs):
            if not vectorstore:
                return None
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            if docs:
                keyword_retriever = BM25Retriever.from_documents(docs)
                keyword_retriever.k = 5
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5]
                )
            else:
                ensemble_retriever = vector_retriever
            if Ranker:
                compressor = FlashRankRerank(top_n=5)
                return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
            return ensemble_retriever

        retrievers['compliance'] = create_hybrid_retriever(db_compliance, docs_compliance)
        retrievers['generation'] = create_hybrid_retriever(db_generation, docs_generation)
        retrievers['whitelist'] = create_hybrid_retriever(db_whitelist, docs_whitelist)
        retrievers['rejected'] = create_hybrid_retriever(db_rejected, docs_rejected)

        for name, retriever in retrievers.items():
            if retriever:
                print(f"✅ '{name}' 리트리버가 성공적으로 생성되었습니다.")
            else:
                print(f"🚨 경고: '{name}' 리트리버를 생성하지 못했습니다 (관련 데이터 파일 부재 추정).")
        print("시스템 초기화 완료.")

    except Exception as e:
        print(f"시스템 초기화 실패: {e}")
        raise e


def process_chat_message(message: str, state: dict) -> dict:
    try:
        if 'step' not in state:
            state['step'] = 'initial'
            state['hasImage'] = False
        if state['step'] == 'initial':
            if 'original_request' not in state:
                state['original_request'] = message
            state['step'] = 'recommend_templates'
            if 'whitelist' not in retrievers or not retrievers['whitelist']:
                state['step'] = 'select_style'
                return {'message': '유사 템플릿 검색 기능이 비활성화 상태입니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:', 'state': state, 'options': ['기본형', '이미지형', '아이템리스트형']}
            similar_docs = retrievers['whitelist'].invoke(state['original_request'])
            if not similar_docs:
                state['step'] = 'select_style'
                return {'message': '유사한 기존 템플릿을 찾지 못했습니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:', 'state': state, 'options': ['기본형', '이미지형', '아이템리스트형']}
            structured_templates = [structure_template_with_llm(doc.page_content) for doc in similar_docs[:3]]
            template_options = [f'템플릿 {i+1} 사용' for i in range(len(similar_docs[:3]))]
            new_creation_options = ['새로 만들기']
            final_options = template_options + new_creation_options
            state['retrieved_similar_templates'] = [doc.page_content for doc in similar_docs[:3]]
            return {
                'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다. 이 템플릿을 사용하시거나 새로 만드시겠어요?', 
                'state': state, 
                'structured_templates': structured_templates, 
                'options': final_options
            }
        elif state['step'] == 'recommend_templates':
            if message.endswith(' 사용'):
                try:
                    template_idx = int(message.split()[1]) - 1
                    selected_template = state['retrieved_similar_templates'][template_idx]
                    state['selected_template_content'] = selected_template
                    state['step'] = 'generate_and_validate'
                    return process_chat_message(message, state)
                except (IndexError, ValueError):
                    pass
            elif message == '새로 만들기':
                state['step'] = 'select_style'
                return {
                    'message': '새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요.',
                    'state': state,
                    'options': ['기본형', '이미지형', '아이템리스트형']
                }
            options = [f'템플릿 {i+1} 사용' for i in range(len(state.get('retrieved_similar_templates',[])))] + ['새로 만들기']
            return {'message': '제시된 옵션 중에서 선택해주세요.', 'state': state, 'options': options}
        elif state.get("step") == "select_style":
            if message in ["기본형", "이미지형", "아이템리스트형"]:
                state["selected_style"] = message
                if message == "이미지형" or "기본형":
                    state["hasImage"] = True
                    state["step"] = "generate_and_validate"
                    return process_chat_message(message, state)
                else:
                    state["step"] = "confirm_image_usage"
                    return {
                        "message": "이미지를 포함하시겠습니까?",
                        "state": state,
                        "options": ["예", "아니오"]
                    }
            else:
                return {
                    "message": "선택하신 스타일이 유효하지 않습니다. '기본형', '이미지형', '아이템리스트형' 중 하나를 선택해주세요.",
                    "state": state,
                    "options": ["기본형", "이미지형", "아이템리스트형"]
                }
        elif state.get("step") == "confirm_image_usage":
            if message in ["예", "아니오"]:
                state["hasImage"] = (message == "예")
                state["step"] = "generate_and_validate"
                return process_chat_message(message, state)
            else:
                return {
                    "message": "잘못된 입력입니다. '예' 또는 '아니오'로만 답해주세요.",
                    "state": state,
                    "options": ["예", "아니오"]
                }
        elif state.get('step') == 'generate_and_validate':
            if 'selected_template_content' in state:
                base_template = state['selected_template_content']
                state['hasImage'] = '(이미지 영역:' in base_template
                del state['selected_template_content']
            else:
                newly_generated = generate_template(
                    request=state['original_request'],
                    style=state.get('selected_style', '기본형')
                )
                param_result = parameterize_template(newly_generated)
                base_template = param_result.get('parameterized_template', newly_generated)
                state['variables_info'] = param_result.get('variables', [])
            state['base_template'] = base_template
            template_draft = fill_template_with_request(template=base_template, request=state['original_request'])
            state['template_draft'] = template_draft
            validation_result = validate_template(template_draft)
            state['validation_result'] = validation_result
            state['correction_attempts'] = 0
            if validation_result['status'] == 'accepted':
                state['step'] = 'completed'
                return process_chat_message(message, state)
            else:
                state['step'] = 'correction'
                return {'message': f'템플릿을 생성했지만 규정 위반이 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result.get("suggestion", "없음")}\n\nAI가 자동으로 수정하겠습니다.', 'state': state}
        elif state['step'] == 'correction':
            if state['correction_attempts'] < MAX_CORRECTION_ATTEMPTS:
                corrected_base_template = correct_template(state)
                state['correction_attempts'] += 1
                validation_result = validate_template(corrected_base_template)
                state["validation_result"] = validation_result
                if validation_result["status"] == "accepted":
                    state['base_template'] = corrected_base_template
                    final_draft = fill_template_with_request(template=corrected_base_template, request=state['original_request'])
                    state['template_draft'] = final_draft
                    state["step"] = "completed"
                    return process_chat_message(message, state)
                else:
                    state['template_draft'] = corrected_base_template
                    return {'message': f'자동 수정 후에도 문제가 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\nAI가 다시 수정하겠습니다.', 'state': state}
            else:
                state['step'] = 'manual_correction'
                return {'message': f'AI 자동 수정이 실패했습니다. 직접 수정하시겠습니까?', 'state': state, 'options': ['포기하기']}
        elif state['step'] == 'manual_correction':
            if message == '포기하기':
                state['step'] = 'initial'
                return {'message': '템플릿 생성을 포기했습니다.', 'state': {'step': 'initial'}}
            else:
                user_corrected_template = message
                validation_result = validate_template(user_corrected_template)
                state['validation_result'] = validation_result
                if validation_result['status'] == 'accepted':
                    state['base_template'] = user_corrected_template
                    final_draft = fill_template_with_request(template=user_corrected_template, request=state['original_request'])
                    state['template_draft'] = final_draft
                    state['step'] = 'completed'
                    return process_chat_message(message, state)
                else:
                    return {'message': f'수정하신 템플릿에도 문제가 있습니다. 다시 수정해주세요.\n\n문제점: {validation_result["reason"]}', 'state': state, 'options': ['포기하기']}
        elif state['step'] == 'completed':
            final_filled_template = state.get("template_draft", "")
            structured_data = structure_template_with_llm(final_filled_template)
            base_template = state.get("base_template", final_filled_template)
            variables = state.get('variables_info', [])
            editable_variables = {"parameterized_template": base_template, "variables": variables} if variables else None
            has_image_flag = state.get('hasImage', False)
            response_message = "✅ 템플릿이 생성되었습니다."
            
            new_state = {'step': 'initial'}
            
            return {
                'message': response_message,
                'state': new_state,
                'template': final_filled_template,
                'structured_template': structured_data.model_dump(),
                'editable_variables': editable_variables,
                'buttons': structured_data.buttons,
                'hasImage': has_image_flag
            }
        return {'message': '알 수 없는 상태입니다. 다시 시도해주세요.', 'state': {'step': 'initial'}}
    except Exception as e:
        print(f"Error in process_chat_message: {e}")
        traceback.print_exc()
        return {'message': f'처리 중 오류가 발생했습니다: {str(e)}', 'state': {'step': 'initial'}}


def fill_template_with_request(template: str, request: str) -> str:
    print(f"템플릿 채우기 시작: 요청='{request}', 템플릿='{template}'")
    variables = re.findall(r'#\{\{([^}]+)\}\}', template)
    if not variables:
        print("템플릿에 채울 변수가 없어 그대로 반환합니다.")
        return template
    variable_names = ", ".join([f"`#{{{{{v}}}}}`" for v in variables])
    prompt = ChatPromptTemplate.from_template(
        '''당신은 주어진 템플릿과 사용자의 구체적인 요청을 결합하여 완성된 메시지를 만드는 전문가입니다.
        # 목표: 사용자의 요청사항을 분석하여, 주어진 템플릿의 각 변수(`#{{변수명}}`)에 가장 적합한 내용을 채워 넣어 완전한 메시지를 생성하세요.
        # 주어진 템플릿:
        ```{template}```
        # 템플릿의 변수 목록: {variable_names}
        # 사용자의 구체적인 요청: "{request}"
        # 지시사항:
        1. 사용자의 요청에서 각 변수에 해당하는 구체적인 정보를 정확히 추출하세요.
        2. 템플릿의 원래 문구와 구조는 절대 변경하지 마세요.
        3. 오직 변수(`#{{...}}`) 부분만 추출한 정보로 대체해야 합니다.
        4. 최종적으로 완성된 템플릿 텍스트만 출력하고, 다른 어떤 설명도 덧붙이지 마세요.
        # 완성된 템플릿:
        '''
    )
    chain = prompt | RunnableLambda(lambda p: call_openai_api(p, model="gpt-4.1", temperature=0.0))
    try:
        filled_template = chain.invoke({
            "template": template,
            "variable_names": variable_names,
            "request": request
        })
        cleaned_template = filled_template.strip().strip('"`')
        print(f"템플릿 채우기 완료: 결과='{cleaned_template}'")
        return cleaned_template
    except Exception as e:
        print(f"Error in fill_template_with_request: {e}")
        return template

def generate_template(request: str, style: str = "기본형") -> str:
    try:
        RULES = {
            "공통": '''
        - GEN-PREVIEW-001 (미리보기 메시지 제한): 채팅방 리스트와 푸시에 노출되는 문구. 한/영 구분 없이 40자까지 입력 가능. 변수 작성 불가.
        - GEN-REVIEW-001 (심사 기본 원칙): 알림톡은 정보통신망법과 카카오 내부 기준에 따라 심사되며, 승인된 템플릿만 발송 가능.
        - GEN-REVIEW-002 (주요 반려 사유): 변수 오류, 과도한 변수(40개 초과) 사용, 변수로만 이루어진 템플릿, 변수가 포함된 버튼명, 변수가 포함된 미리보기 메시지 설정 시 반려됨.
        - GEN-INFO-DEF-001 (정보성 메시지의 정의): 고객의 요청에 의한 1회성 정보, 거래 확인, 계약 변경 안내 등이 포함됨. 부수적으로 광고가 포함되면 전체가 광고성 정보로 간주됨.
        - GEN-SERVICE-STD-001 (알림톡 서비스 기준): 알림톡은 수신자에게 반드시 전달되어야 하는 '정형화된 정보성' 메시지에 한함.
        - GEN-BLACKLIST-001 (블랙리스트 - 포인트/쿠폰): 수신자 동의 없는 포인트 적립/소멸 메시지, 유효기간이 매우 짧은 쿠폰 등은 발송 불가.
        - GEN-BLACKLIST-002 (블랙리스트 - 사용자 행동 기반): 장바구니 상품 안내, 클릭했던 상품 안내, 생일 축하 메시지, 앱 다운로드 유도 등은 발송 불가.
        - GEN-GUIDE-001 (정보성/광고성 판단 기준): 특가/할인 상품 안내, 프로모션 또는 이벤트가 혼재된 경우는 광고성 메시지로 판단됨.
        ''',
            "기본형": {
                "규칙": '''
        - GEN-TYPE-001 (기본형 특징 및 제한): 고객에게 반드시 전달되어야 하는 정보성 메시지. 한/영 구분 없이 1,000자까지 입력 가능하며, 개인화된 텍스트 영역은 #{변수}로 작성.
        - GEN-TYPE-002 (부가 정보형 특징 및 제한): 고정적인 부가 정보를 본문 하단에 안내. 최대 500자, 변수 사용 불가, URL 포함 가능. 본문과 합쳐 총 1,000자 초과 불가.
        - GEN-TYPE-003 (채널추가형 특징 및 제한): 비광고성 메시지 하단에 채널 추가 유도. 안내 멘트는 최대 80자, 변수/URL 포함 불가.
        ''',
                "스타일 가이드": '''
        # 스타일 설명: 텍스트 중심으로 정보를 전달하는 가장 기본적인 템플릿입니다. 간결하고 직관적인 구성으로 공지, 안내, 상태 변경 등 명확한 내용 전달에 사용됩니다.
        # 대표 예시 1 (서비스 완료 안내)
        안녕하세요, #{수신자명}님. 요청하신 #{서비스} 처리가 완료되었습니다. 자세한 내용은 아래 버튼을 통해 확인해주세요.
        # 대표 예시 2 (예약 리마인드)
        안녕하세요, #{수신자명}님. 내일(#{예약일시})에 예약하신 서비스가 예정되어 있습니다. 잊지 말고 방문해주세요.
        '''
            },
            "이미지형": {
                "규칙": '''
        - GEN-STYLE-001 (이미지형 특징 및 제한): 포맷화된 정보성 메시지를 시각적으로 안내. 광고성 내용 포함 불가. 템플릿 당 하나의 고정된 이미지만 사용 가능.
        - GEN-STYLE-002 (이미지형 제작 가이드 - 사이즈): 권장 사이즈는 800x400px (JPG, PNG), 최대 500KB.
        - GEN-STYLE-009 (이미지 저작권 및 내용 제한): 타인의 지적재산권, 초상권을 침해하는 이미지, 본문과 관련 없는 이미지, 광고성 이미지는 절대 사용 불가.
        ''',
                "스타일 가이드": '''
        # 스타일 설명: 시각적 요소를 활용하여 사용자의 시선을 끌고 정보를 효과적으로 전달하는 템플릿입니다. 상품 홍보, 이벤트 안내 등 시각적 임팩트가 중요할 때 사용됩니다.
        # 대표 예시 1 (신상품 출시)
        (이미지 영역: 새로 출시된 화장품 라인업)
        '''
            }
        }
        generation_rules = retrievers.get('generation').invoke(request)
        formatted_rules = "\n".join([f"- {doc.metadata.get('rule_id', 'Unknown')}: {doc.page_content}" for doc in generation_rules])
        
        prompt = ChatPromptTemplate.from_template(
            '''You are a highly precise, rule-based Kakao Alimtalk Template Generation Bot. Your sole mission is to generate a perfect template draft that strictly adheres to all user requests, style guides, and provided rules.

### Final Goal:
Create a ready-to-use Alimtalk template draft that reflects the user's request, utilizes the features of the selected style, and **complies with every single provided rule without exception.**

### Input Information:
- **User's Original Request:** "{request}"
- **Style to Apply:** {style}
- **Style Guide:** {style_guide}
- **Absolute Rules to Follow:** {rules}

### Execution Steps:
1.  **Analyze Request:** Meticulously analyze the user's original request to identify the core purpose and required information for the template.
2.  **Apply Style:** Refer to the style guide's description and examples to determine the overall structure and tone & manner.
3.  **Ensure Compliance:** Review **every rule** in the `Absolute Rules to Follow` list. Ensure the generated template does not violate any of them. Pay special attention to variable usage rules (e.g., variable names in Korean, no variables in button names).
4.  **Parameterize:** Identify specific, changeable information (e.g., customer names, dates, product names, amounts) and convert it into the `#{{variable_name}}` format. The variable name must be a concise and clear Korean word representing the information (e.g., `#{{고객명}}`, `#{{주문번호}}`).
5.  **Output Format:** Your **only** output must be the raw text of the generated template. Do not include any introductory phrases, explanations, markdown code blocks (```), or any text other than the template itself.

### Example:
---
**Input Example:**
- User's Original Request: "A message to inform the customer that their order has been successfully received, and provide the order number and estimated delivery date. Include a button to check order details on the website."
- Style to Apply: 기본형
- Style Guide: # 스타일 설명: 텍스트 중심으로 정보를 전달하는 가장 기본적인 템플릿입니다.
- Absolute Rules to Follow:
    - GEN-TYPE-001: Informational message, up to 1,000 characters, personalized parts should be variables.
    - GEN-REVIEW-002: No variables in button names.

**Correct Output Example:**
안녕하세요, #{{고객명}}님.
주문이 성공적으로 접수되었습니다.

주문 번호: #{{주문번호}}
예상 배송일: #{{예상배송일}}

주문 상세는 아래 버튼을 통해 확인해주세요.

[웹사이트] 주문 상세 확인하기
---

### Generated Template Draft:
'''
        )

        chain = prompt | RunnableLambda(lambda p: call_openai_api(p, model="gpt-4.1", temperature=0.0))
        template = chain.invoke({
            "request": request,
            "style": style,
            "style_guide": RULES.get(style, {}).get("스타일 가이드", ""),
            "rules": f'{RULES["공통"]}\n{RULES.get(style, {}).get("규칙", "")}\n관련 규칙:\n{formatted_rules}'
        })
        return template.strip()
    except Exception as e:
        print(f"Error in generate_template: {e}")
        return f"템플릿 생성 중 오류 발생: {request}"


def validate_template(template: str) -> Dict:
    parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
    relevant_rules = retrievers['compliance'].invoke(template)
    formatted_rules = "\n".join([f"- {doc.metadata.get('source', 'content')}: {doc.page_content}" for doc in relevant_rules])
    prompt = ChatPromptTemplate.from_template(
        '''당신은 카카오 알림톡 심사 가이드라인을 완벽하게 숙지한 AI 심사관입니다.
        주어진 템플릿이 모든 규칙을 준수하는지 검사하고, 결과를 JSON 형식으로 반환하세요.
        # 검사할 템플릿:
        ```{template}```
        # 주요 심사 규칙:
        {rules}
        # 지시사항:
        1. 템플릿이 모든 규칙을 준수하면 `status`를 "accepted"로 설정합니다.
        2. 규칙 위반 사항이 하나라도 발견되면 `status`를 "rejected"로 설정합니다.
        3. "rejected"인 경우, `reason`에 어떤 규칙을 위반했는지 명확하고 상세하게 설명합니다.
        4. `evidence` 필드에는 위반의 근거가 된 규칙의 `content`를 정확히 기재합니다.
        5. 위반 사항을 해결할 수 있는 구체적인 `suggestion`을 제공합니다.
        6. 최종 결과는 반드시 지정된 JSON 형식으로만 출력해야 합니다.
        # 심사 결과 (JSON):
        {format_instructions}
        '''
    )
    chain = prompt | RunnableLambda(lambda p: call_openai_api_for_json(p, model="gpt-4o")) | parser
    try:
        result = chain.invoke({
            "template": template,
            "rules": formatted_rules,
            "format_instructions": parser.get_format_instructions()
        })
        # [수정] 결과에 'status' 키가 없는 경우를 대비한 방어 코드
        if 'status' not in result:
            return {"status": "error", "reason": "검증 결과 파싱 중 오류 발생"}
        return result
    except Exception as e:
        print(f"Error during validation: {e}")
        return {"status": "error", "reason": f"검증 중 예외 발생: {e}"}


def correct_template(state: dict) -> str:
    validation_result = state['validation_result']
    original_template = state['template_draft']
    prompt = ChatPromptTemplate.from_template(
        '''당신은 템플릿의 문제점을 분석하고 수정하는 AI 전문가입니다.
        주어진 원본 템플릿과 반려 사유를 바탕으로, 모든 문제를 해결한 새로운 템플릿을 제안하세요.
        # 원본 템플릿:
        ```{original_template}```
        # 반려 사유 및 수정 제안:
        - 이유: {reason}
        - 근거: {evidence}
        - 제안: {suggestion}
        # 지시사항:
        1. 반려 사유를 명확히 이해하고, 어떤 부분을 수정해야 할지 파악합니다.
        2. 수정 제안을 참고하여 템플릿을 개선합니다.
        3. 원본 템플릿의 의도는 최대한 유지하면서 문제점만 해결해야 합니다.
        4. 최종 결과물은 수정된 템플릿 텍스트만 출력해야 합니다. 다른 어떤 설명도 추가하지 마세요.
        # 수정된 템플릿:
        '''
    )
    chain = prompt | RunnableLambda(lambda p: call_openai_api(p, model="gpt-4o"))
    try:
        corrected_template = chain.invoke({
            "original_template": original_template,
            "reason": validation_result.get("reason", ""),
            "evidence": validation_result.get("evidence", ""),
            "suggestion": validation_result.get("suggestion", "")
        })
        return corrected_template.strip()
    except Exception as e:
        print(f"Error during correction: {e}")
        return original_template
