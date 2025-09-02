import os
import json
import re
import traceback
from typing import TypedDict, List, Optional, Dict
import sys

# Pydantic 및 LangChain 호환성을 위한 임포트
from pydantic import BaseModel, Field, PrivateAttr

# LangChain 및 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

# FlashRank 임포트
try:
    from flashrank import Ranker, RerankRequest
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from langchain_core.callbacks.base import Callbacks
except ImportError:
    print("FlashRank 또는 관련 모듈을 찾을 수 없습니다.")
    BaseDocumentCompressor = object
    Ranker = None

# --- 설정 및 모델 정의 ---
MAX_CORRECTION_ATTEMPTS = 3

# --- RA-HyDE (Retrieval-Augmented HyDE) 구현 클래스 ---
class ManualHydeRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    llm: Runnable
    prompt: ChatPromptTemplate

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        few_shot_docs = self.base_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child()
        )
        context = "\n\n---\n\n".join([doc.page_content for doc in few_shot_docs[:2]])
        generation_chain = self.prompt | self.llm | StrOutputParser()
        hypothetical_document = generation_chain.invoke(
            {"question": query, "context": context},
            config={"callbacks": run_manager.get_child()}
        )
        return self.base_retriever.get_relevant_documents(
            hypothetical_document, callbacks=run_manager.get_child()
        )


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
    status: str = Field(description="템플릿의 최종 상태 (예: 'accepted', 'rejected')")
    reason: str = Field(description="상세한 판단 이유 (반드시 한국어로 작성)")
    evidence: Optional[str] = Field(None, description="판단 근거가 된 규칙의 rule_id (쉼표로 구분)")
    suggestion: Optional[str] = Field(None, description="템플릿 개선을 위한 구체적인 제안 (반드시 한국어로 작성)")
    revised_template: Optional[str] = Field(None, description="규정에 맞게 수정된 템플릿 예시")

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
    name: str = Field(description="추출된 변수의 한글 이름 (예: 매장명, 폐점일자). `#{}`에 들어갈 부분입니다.")
    original_value: str = Field(description="원본 텍스트에서 추출된 실제 값")
    description: str = Field(description="해당 변수에 대한 간단한 한글 설명 (사용자가 이해하기 쉽도록)")

class ParameterizedResult(BaseModel):
    parameterized_template: str = Field(description="특정 정보가 #{변수명}으로 대체된 최종 템플릿")
    variables: List[Variable] = Field(description="추출된 변수들의 목록")

# --- 전역 변수 및 헬퍼 함수 ---
llm = None
retrievers = {}
approved_templates = []
rejected_templates = []

# 스타일별 예시 템플릿 정의 (아이콘 마커 추가)
STYLE_EXAMPLES = {
    "기본형": [
        "회원 안내사항 전달\n안녕하세요, #{수신자}님!\n#{발신 스페이스}의 회원이 되신 것을 환영합니다.\n\n🔔 신규 회원 안내 사항 🔔\n#{안내사항}\n\n버튼명",
    ],
    "이미지형": [
        "[ICON=USER]\n회원 등급 변경\n안녕하세요, #{수신자명}님.\n#{발신 스페이스}입니다.\n\n고객님의 회원 등급이 변경되었습니다.\n- 기존 등급: #{기존 등급}\n- 변경 등급: #{변경 등급}\n\n앞으로도 많은 이용 부탁드립니다.\n버튼명",
    ],
    "아이템리스트형": [
        "[ICON=BARCODE]\n참가코드 안내\n안녕하세요, #{발신 스페이스}입니다.\n\n예정된 온라인 교육의 참가코드가 아래와 같이 발송되었습니다.\n- 교육일자: #{교육일자}\n- 교육시간: #{교육시간}\n- 참가코드: #{참가코드}\n\n자세한 문의는 #{문의번호}로 연락해 주세요.\n감사합니다.\n버튼명",
    ]
}

def detect_template_style(template_content: str) -> str:
    """템플릿 내용을 분석하여 스타일을 자동으로 감지합니다."""
    if '[ICON=' in template_content:
        return '이미지형'
    elif '▶' in template_content or ('- ' in template_content and ':' in template_content):
        return '아이템리스트형'
    else:
        return '기본형'

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

def parameterize_template(template_string: str) -> Dict:
    parser = JsonOutputParser(pydantic_object=ParameterizedResult)
    prompt = ChatPromptTemplate.from_template(
        """당신은 주어진 텍스트를 재사용 가능한 템플릿으로 변환하는 전문가입니다.
        주어진 텍스트에서 고유명사, 날짜, 장소, 숫자 등 구체적이고 바뀔 수 있는 정보들을 식별하여, 의미 있는 한글 변수명으로 대체해주세요.
        
        # 지시사항
        1. 텍스트의 핵심 정보(누가, 언제, 어디서, 무엇을, 어떻게 등)를 파악합니다.
        2. 파악된 정보를 `#{{변수명}}` 형태로 대체하여 재사용 가능한 템플릿을 생성합니다. 변수명은 사용자가 이해하기 쉬운 한글로 작성하세요.
        3. 원본 값과 변수명, 그리고 각 변수에 대한 설명을 포함하는 변수 목록을 생성합니다.
        4. **모든 설명과 변수명은 반드시 한국어로 작성해야 합니다.**
        5. 최종 결과를 지정된 JSON 형식으로만 출력해야 합니다. 그 외의 설명은 절대 추가하지 마세요.

        # 원본 텍스트:
        {original_text}

        # 출력 형식 (JSON):
        {format_instructions}
        """
    )
    chain = prompt | llm | parser
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

def create_hybrid_retriever(vectorstore, docs, llm, embeddings, hyde_prompt):
    if not vectorstore:
        return None
    base_vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    manual_hyde_retriever = ManualHydeRetriever(
        base_retriever=base_vector_retriever,
        llm=llm,
        prompt=hyde_prompt
    )
    if docs:
        keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever.k = 10
        ensemble_retriever = EnsembleRetriever(
            retrievers=[manual_hyde_retriever, keyword_retriever], weights=[0.6, 0.4]
        )
    else:
        ensemble_retriever = manual_hyde_retriever
    if Ranker:
        compressor = FlashRankRerank(top_n=5)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
    return ensemble_retriever

def initialize_system():
    global llm, retrievers, approved_templates, rejected_templates
    if llm is not None:
        return

    print("서버 시작: 시스템 초기화를 진행합니다...")
    try:
        data_dir = 'data'
        vector_db_path = "vector_db"
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
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
            if docs:
                return Chroma.from_documents(
                    docs, embeddings, collection_name=name,
                    persist_directory=vector_db_path, client_settings=client_settings
                )
            return None

        db_compliance = create_db("compliance_rules", docs_compliance)
        db_generation = create_db("generation_rules", docs_generation)
        db_whitelist = create_db("whitelist_templates", docs_whitelist)
        db_rejected = create_db("rejected_templates", docs_rejected)

        hyde_prompt_base = """당신은 전문가입니다. 아래 예시를 참고하여, 사용자의 요청에 가장 이상적인 {doc_type}을(를) 한글로 작성해주세요.
# 예시: {context}
# 사용자 요청: {question}
# 이상적인 {doc_type}:"""

        base_prompt_template = ChatPromptTemplate.from_template(hyde_prompt_base)
        prompts = {
            'compliance': base_prompt_template.partial(doc_type="검수 규정 문서"),
            'generation': base_prompt_template.partial(doc_type="템플릿 생성 가이드라인"),
            'whitelist': base_prompt_template.partial(doc_type="승인 템플릿 예시"),
            'rejected': base_prompt_template.partial(doc_type="반려 사례")
        }

        dbs = {
            'compliance': (db_compliance, docs_compliance),
            'generation': (db_generation, docs_generation),
            'whitelist': (db_whitelist, docs_whitelist),
            'rejected': (db_rejected, docs_rejected)
        }

        for name, (db, docs) in dbs.items():
            if db:
                retrievers[name] = create_hybrid_retriever(db, docs, llm, embeddings, prompts[name])
                print(f"✅ '{name}' 리트리버가 성공적으로 생성되었습니다.")
            else:
                print(f"🚨 경고: '{name}' 리트리버를 생성하지 못했습니다.")

        print("시스템 초기화 완료.")
    except Exception as e:
        print(f"시스템 초기화 실패: {e}")
        raise e
    
def fill_template_with_request(template_structure: str, user_request: str) -> str:
    """
    주어진 템플릿 구조(틀)와 사용자 요청을 바탕으로, LLM을 사용하여 변수를 채워넣은 완성된 템플릿을 생성합니다.
    특히, #{제목}과 같은 요약 변수는 사용자 요청의 핵심을 파악하여 생성합니다.
    """
    prompt = ChatPromptTemplate.from_template(
        """당신은 지시를 매우 잘 따르는 AI 어시스턴트입니다.
        주어진 '템플릿 구조'의 변수(`#{{...}}`)들을 '사용자 요청'에서 찾을 수 있는 정보로 채워넣어, 완성된 알림톡 메시지를 만드세요.

        # 템플릿 구조:
        {template_structure}

        # 사용자 요청:
        {user_request}

        # 지시사항:
        1. '사용자 요청'의 핵심 내용을 파악하여 '템플릿 구조'의 각 변수에 가장 적절한 값을 채워넣으세요.
        2. 특히 `#{{제목}}`이나 `#{{요약}}`과 같은 변수는, 사용자 요청의 핵심 내용을 한 문장으로 요약하여 채워넣어야 합니다. (예: "주간 회의 참석 안내", "3월 급여명세서 발송")
        3. 만약 '사용자 요청'에서 특정 변수에 대한 정보를 찾을 수 없다면, 해당 변수는 `#{{변수명}}` 형태로 그대로 남겨두세요.
        4. 최종 결과물은 오직 완성된 템플릿 텍스트여야 합니다. 다른 설명이나 주석은 절대 추가하지 마세요.

        # 완성된 템플릿:
        """
    )
    chain = prompt | llm | StrOutputParser()
    filled_template = chain.invoke({
        "template_structure": template_structure,
        "user_request": user_request
    })
    return filled_template.strip()

def process_chat_message(message: str, state: dict) -> dict:
    try:
        current_step = state.get('step', 'initial')
        print(f"--- Processing Step: {current_step}, Message: {message} ---")

        while True:
            print(f"Executing step: {current_step}")
            next_step = None

            if current_step == 'initial':
                state['original_request'] = message
                next_step = 'recommend_templates'

            elif current_step == 'recommend_templates':
                if 'whitelist' not in retrievers or not retrievers['whitelist']:
                    next_step = 'ask_for_style'
                else:
                    similar_docs = retrievers['whitelist'].invoke(state['original_request'])
                    if not similar_docs:
                        next_step = 'ask_for_style'
                    else:
                        templates = [doc.page_content for doc in similar_docs[:3]]
                        template_options = [f"템플릿 {i+1}" for i in range(len(templates))]
                        
                        state['recommended_templates'] = templates
                        state['step'] = 'select_or_create'

                        templates_data = []
                        for template in templates:
                            detected_style = detect_template_style(template)
                            templates_data.append({
                                'content': template,
                                'style': detected_style
                            })

                        return {
                            'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다. 아래에서 선택하시거나 "신규 생성"을 선택해주세요.\n(버튼 클릭 시 우측에 미리보기가 표시됩니다)',
                            'state': state,
                            'options': template_options + ['신규 생성'],
                            'templates_data': templates_data
                        }
            
            elif current_step == 'select_or_create':
                if message.startswith('템플릿'):
                    try:
                        template_idx = int(message.split(' ')[1]) - 1
                        if 0 <= template_idx < len(state.get('recommended_templates', [])):
                            state['selected_template_structure'] = state['recommended_templates'][template_idx]
                            next_step = 'fill_template'
                        else:
                            return {'message': '잘못된 템플릿 번호입니다. 다시 선택해주세요.', 'state': state}
                    except (ValueError, IndexError):
                        next_step = 'ask_for_style'
                
                elif message == '신규 생성':
                    next_step = 'ask_for_style'
                else:
                    return {'message': '알 수 없는 선택입니다. 다시 시도해주세요.', 'state': state}

            elif current_step == 'fill_template':
                print("사용자 요청으로 템플릿 채우는 중...")
                filled_draft = fill_template_with_request(
                    template_structure=state['selected_template_structure'],
                    user_request=state['original_request']
                )
                state['template_draft'] = filled_draft
                state['selected_style'] = detect_template_style(filled_draft)
                next_step = 'validation'

            elif current_step == 'ask_for_style':
                options = ['기본형', '이미지형', '아이템리스트형']
                templates_data = []
                for style in options:
                    example_template = STYLE_EXAMPLES.get(style, [""])[0] 
                    templates_data.append({
                        'content': example_template,
                        'style': style
                    })
                state['step'] = 'process_style_selection'
                return {
                    'message': '어떤 스타일의 템플릿을 원하시나요? 각 스타일의 미리보기를 확인하고 선택해주세요.',
                    'state': state,
                    'options': options,
                    'templates_data': templates_data
                }

            elif current_step == 'process_style_selection':
                if message in ['기본형', '이미지형', '아이템리스트형']:
                    state['selected_style'] = message
                    next_step = 'generate_template'
                else:
                    state['step'] = 'ask_for_style'
                    return {'message': '올바른 스타일을 선택해주세요.', 'state': state}

            elif current_step == 'generate_template':
                print(f"템플릿 생성 중... 요청: {state['original_request']}, 스타일: {state['selected_style']}")
                template_draft = generate_template(state['original_request'], state['selected_style'])
                state['template_draft'] = template_draft
                next_step = 'validation'

            elif current_step == 'validation':
                print("템플릿 검증 중...")
                validation_result = validate_template(state['template_draft'])
                state['validation_result'] = validation_result
                
                if validation_result.get('status') == 'accepted':
                    next_step = 'completed'
                else:
                    state['correction_attempts'] = state.get('correction_attempts', 0)
                    next_step = 'correction'

            elif current_step == 'correction':
                attempts = state.get('correction_attempts', 0)
                if attempts < MAX_CORRECTION_ATTEMPTS:
                    print(f"자동 수정 시도 중... ({attempts + 1}/{MAX_CORRECTION_ATTEMPTS})")
                    corrected_template = correct_template(state)
                    state['template_draft'] = corrected_template
                    state['correction_attempts'] = attempts + 1
                    next_step = 'validation'
                else:
                    state['step'] = 'manual_correction'
                    return {
                        'message': f'AI 자동 수정이 {MAX_CORRECTION_ATTEMPTS}회 모두 실패했습니다.\n\n마지막 시도 결과:\n{state["validation_result"]["reason"]}\n\n직접 수정해주시거나 "포기하기"를 선택해주세요.\n\n현재 템플릿:\n{state["template_draft"]}',
                        'state': state,
                        'options': ['포기하기'],
                        'template_data': {
                            'content': state["template_draft"],
                            'style': state.get("selected_style", "기본형")
                        }
                    }

            elif current_step == 'manual_correction':
                if message == '포기하기':
                    return {
                        'message': '템플릿 생성을 포기했습니다. 새로운 요청을 해주세요.',
                        'state': {'step': 'initial'}
                    }
                else:
                    state['template_draft'] = message
                    next_step = 'validation'

            elif current_step == 'completed':
                final_template = state.get("template_draft", "")
                final_style = state.get("selected_style", "기본형")
                parameterized_result = parameterize_template(final_template)
                
                return {
                    'message': '✅ 템플릿이 성공적으로 생성되었습니다!',
                    'state': {'step': 'initial'},
                    'template': final_template,
                    'editable_variables': parameterized_result,
                    'template_data': {
                        'content': parameterized_result.get('parameterized_template', final_template),
                        'style': final_style
                    }
                }

            else:
                return {'message': '알 수 없는 상태입니다. 다시 시도해주세요.', 'state': {'step': 'initial'}}

            if next_step:
                current_step = next_step
            else:
                break
        
        return {'message': '처리가 완료되었지만 반환할 메시지가 없습니다.', 'state': state}

    except Exception as e:
        print(f"Error in process_chat_message: {e}")
        traceback.print_exc()
        return {'message': f'처리 중 오류가 발생했습니다: {str(e)}', 'state': {'step': 'initial'}}

def generate_template(request: str, style: str = "기본형") -> str:
    try:
        rules = "사용자의 요청에 맞춰 정보성 템플릿을 생성하세요."
        if 'generation' in retrievers and retrievers['generation']:
            docs = retrievers['generation'].invoke(request)
            rules = "\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_template(
            """당신은 알림톡 템플릿 생성 전문가입니다. 사용자의 요청에 따라 적절한 알림톡 템플릿을 생성해주세요.
            # 사용자 요청: {request}
            # 선택된 스타일: {style}
            # 생성 규칙: {rules}
            # 지시사항: 템플릿은 제목, 본문, 버튼 텍스트로 구성되어야 하며, 각 줄은 개행으로 구분됩니다. 템플릿 텍스트만 출력하세요."""
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"request": request, "style": style, "rules": rules}).strip()
    except Exception as e:
        print(f"Error in generate_template: {e}")
        return "템플릿 생성 중 오류가 발생했습니다.\n다시 시도해주세요.\n확인"

def validate_template(draft: str) -> dict:
    try:
        rules, rejections = "광고성 문구 금지", "광고성 문구 포함된 사례"
        if 'compliance' in retrievers and retrievers['compliance']:
            docs = retrievers['compliance'].invoke(draft)
            rules = "\n".join([f"[ID: {doc.metadata.get('rule_id', 'N/A')}] {doc.page_content}" for doc in docs])
        if 'rejected' in retrievers and retrievers['rejected']:
            docs = retrievers['rejected'].invoke(draft)
            rejections = "\n".join([doc.page_content for doc in docs])

        parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
        prompt = ChatPromptTemplate.from_template(
            """당신은 알림톡 템플릿의 규정 준수 여부를 판단하는 한국어 전문가입니다.
            템플릿 초안을 검토하고, 규정 위반 여부를 JSON 형식으로 분석해주세요.

            # 중요 지시사항:
            - 최종 분석 결과(reason, suggestion 등)는 **반드시 한국어로 작성**해야 합니다. 절대로 영어를 사용하지 마세요.
            - `status`는 'accepted'(승인) 또는 'rejected'(반려) 중 하나여야 합니다.

            # 검토할 템플릿 초안:
            {draft}

            # 준수해야 할 규칙들:
            {rules}

            # 과거 반려된 템플릿 사례들:
            {rejections}

            # 출력 형식 (JSON):
            {format_instructions}"""
        )
        chain = prompt | llm | parser
        return chain.invoke({
            "draft": draft, "rules": rules, "rejections": rejections,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        print(f"Error in validate_template: {e}")
        return {"status": "accepted", "reason": "검증 중 오류 발생. 자동 승인 처리됩니다.", "suggestion": None, "revised_template": draft}

def correct_template(state: dict) -> str:
    try:
        attempts = state.get('correction_attempts', 0)
        if attempts <= 1:
            instruction = "3. 광고성 문구를 제거하거나, 정보성 내용으로 순화하는 등, 제안된 방향에 맞게 템플릿을 수정하세요."
        elif attempts == 2:
            instruction = "3. **(2차 수정)** 아직도 문제가 있습니다. 이번에는 '쿠폰', '할인', '이벤트', '특가'와 같은 명백한 광고성 단어를 사용하지 말고, 정보 전달에만 집중하세요."
        else:
            instruction = """3. **(최종 수정: 관점 전환)** 여전히 광고성으로 보입니다. 이것이 마지막 시도입니다.
            - **관점 전환:** 메시지의 주체를 '우리(사업자)'에서 '고객님'으로 완전히 바꾸세요.
            - **목적 변경:** '판매'나 '방문 유도'가 아니라, '고객님이 과거에 동의한 내용에 따라 고객님의 권리(혜택) 정보를 안내'하는 것으로 목적을 재정의하세요."""

        correction_prompt_template = """당신은 지적된 문제점을 해결하여 더 나은 대안을 제시하는 전문 카피라이터입니다.
        당신의 유일한 임무는 아래 지시사항에 따라 **수정된 템플릿 초안 하나만**을 생성하는 것입니다. 초안 외에 다른 설명은 절대로 덧붙이지 마세요.
        
        # 원래 사용자 요청: {original_request}
        # 이전에 제안했던 템플릿 (반려됨): {rejected_draft}
        # 반려 사유 및 개선 제안: {rejection_reason}
        
        # 지시사항
        1. '반려 사유 및 개선 제안'을 완벽하게 이해하고, 지적된 모든 문제점을 해결하세요.
        2. '원래 사용자 요청'의 핵심 의도는 유지해야 합니다.
        {dynamic_instruction}
        
        # 수정된 템플릿 초안 (오직 템플릿 텍스트만 출력):
        """
        correction_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
        correction_prompt = correction_prompt.partial(dynamic_instruction=instruction)
        correction_chain = correction_prompt | llm | StrOutputParser()

        rejection_reason = state['validation_result']['reason']
        if state['validation_result'].get('suggestion'):
            rejection_reason += "\n개선 제안: " + state['validation_result']['suggestion']

        new_draft = correction_chain.invoke({
            "original_request": state['original_request'],
            "rejected_draft": state['template_draft'],
            "rejection_reason": rejection_reason
        })
        return new_draft.strip()
    except Exception as e:
        print(f"Error in correct_template: {e}")
        return state.get('template_draft', '수정 중 오류가 발생했습니다.')
