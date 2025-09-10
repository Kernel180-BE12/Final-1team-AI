# chatbot_logic.py

import os
import json
import re
from typing import TypedDict, List, Optional, Dict
import sys
import traceback

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
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.callbacks.base import Callbacks


# FlashRank 임포트
try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    print("FlashRank 또는 관련 모듈을 찾을 수 없습니다. Reranking 기능이 비활성화됩니다.")
    Ranker = None

# --- 설정 및 모델 정의 ---
MAX_CORRECTION_ATTEMPTS = 3

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

# --- 전역 변수 및 헬퍼 함수 ---
llm = None
retrievers = {}
approved_templates = []
rejected_templates = []

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
    """LLM을 사용해 템플릿 텍스트를 구조화된 객체로 변환합니다."""
    parser = JsonOutputParser(pydantic_object=StructuredTemplate)

    prompt = ChatPromptTemplate.from_template(
        """당신은 주어진 텍스트를 분석하여 핵심 구성 요소로 구조화하는 전문가입니다.
        # 지시사항:
        1. 텍스트의 핵심 의도를 'title'로 추출합니다. 길이는 짧고 간결하게.
        2. 텍스트의 가장 마지막 줄에 있는 버튼 정보를 분석하여 'buttons' 리스트를 생성합니다.
        3. 버튼은 최대 2개까지 생성할 수 있으며, 없으면 빈 리스트 `[]`를 반환합니다.
        4. 각 버튼은 ['버튼종류', '버튼이름'] 형식의 튜플이어야 합니다. 버튼 종류는 내용에 맞게 '웹사이트', '앱링크', '전화하기' 등으로 추론하세요.
        5. 제목과 버튼을 제외한 나머지 모든 내용을 'body'로 추출합니다.
        6. 최종 결과를 지정된 JSON 형식으로만 출력해야 합니다.

        # 원본 텍스트:
        {raw_text}

        # 출력 형식 (JSON):
        {format_instructions}
        """
    )

    chain = prompt | llm | parser
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

def render_template_from_structured(data: StructuredTemplate) -> StructuredTemplate:
    """구조화된 데이터를 받아 StructuredTemplate 객체를 반환합니다."""
    return data

def parameterize_template(template_string: str) -> Dict:
    parser = JsonOutputParser(pydantic_object=ParameterizedResult)
    prompt = ChatPromptTemplate.from_template(
        """당신은 주어진 텍스트를 재사용 가능한 템플릿으로 변환하는 전문가입니다.
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

def initialize_system():
    global llm, retrievers, approved_templates, rejected_templates
    if llm is not None:
        return
        
    print("서버 시작: 시스템 초기화를 진행합니다...")
    try:
        data_dir = 'data'
        vector_db_path = "vector_db"
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
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

            if os.path.exists(vector_db_path):
                try:
                    print(f"🔍 '{vector_db_path}'에서 기존 '{name}' 컬렉션을 불러옵니다...")
                    db = Chroma(
                        collection_name=name,
                        persist_directory=vector_db_path,
                        embedding_function=embeddings,
                        client_settings=client_settings
                    )
                    if db._collection.count() > 0:
                        print(f"✅ '{name}' 컬렉션을 성공적으로 불러왔습니다. (항목 수: {db._collection.count()})")
                        return db
                    else:
                        print(f"🤔 '{name}' 컬렉션은 존재하지만 비어있습니다. 새로 생성합니다.")
                except Exception as e:
                    print(f"⚠️ 기존 DB를 불러오는 중 오류 발생({e}). DB를 새로 생성합니다.")
                    pass

            print(f"✨ '{name}' 컬렉션을 새로 생성하고 디스크에 저장합니다...")
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

# chatbot_logic.py

# ... (파일의 다른 부분은 모두 그대로 유지) ...

def process_chat_message(message: str, state: dict) -> dict:
    """채팅 메시지 처리 - hasImage 및 템플릿 추천 로직 적용"""
    try:
        if 'step' not in state:
            state['step'] = 'initial'
            state['hasImage'] = False # 상태 초기화

        if state['step'] == 'initial':
            if 'original_request' not in state:
                state['original_request'] = message
            
            state['step'] = 'recommend_templates'
            
            # 유사 템플릿 검색 기능이 비활성화 상태이거나, retriever가 없는 경우 바로 스타일 선택으로
            if 'whitelist' not in retrievers or not retrievers['whitelist']:
                state['step'] = 'select_style'
                return {'message': '유사 템플릿 검색 기능이 비활성화 상태입니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:', 'state': state, 'options': ['기본형', '이미지형', '아이템리스트형']}

            # 유사 템플릿 검색
            similar_docs = retrievers['whitelist'].invoke(state['original_request'])
            
            # 유사 템플릿이 없는 경우 바로 스타일 선택으로
            if not similar_docs:
                state['step'] = 'select_style'
                return {'message': '유사한 기존 템플릿을 찾지 못했습니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:', 'state': state, 'options': ['기본형', '이미지형', '아이템리스트형']}
            
            # 유사 템플릿이 있는 경우, 추천 메시지 표시
            structured_templates = [render_template_from_structured(structure_template_with_llm(doc.page_content)) for doc in similar_docs[:3]]
            
            template_options = [f'템플릿 {i+1} 사용' for i in range(len(similar_docs[:3]))]
            new_creation_options = ['새로 만들기']
            final_options = template_options + new_creation_options

            # state에 검색된 템플릿 저장
            state['retrieved_similar_templates'] = [doc.page_content for doc in similar_docs[:3]]

            return {
                'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다. 이 템플릿을 사용하시거나 새로 만드시겠어요?', 
                'state': state, 
                'structured_templates': structured_templates, 
                'options': final_options
            }

        elif state['step'] == 'recommend_templates':
            if message.endswith(' 사용'): # 예: '템플릿 1 사용'
                try:
                    template_idx = int(message.split()[1]) - 1
                    selected_template = state['retrieved_similar_templates'][template_idx]
                    
                    state['selected_template_content'] = selected_template
                    state['step'] = 'generate_and_validate'
                    return process_chat_message(message, state)
                except (IndexError, ValueError):
                    # 잘못된 입력을 한 경우
                    pass

            elif message == '새로 만들기':
                state['step'] = 'select_style'
                return {
                    'message': '새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요.',
                    'state': state,
                    'options': ['기본형', '이미지형', '아이템리스트형']
                }

            # 유효하지 않은 옵션 처리
            options = [f'템플릿 {i+1} 사용' for i in range(len(state.get('retrieved_similar_templates',[])))] + ['새로 만들기']
            return {'message': '제시된 옵션 중에서 선택해주세요.', 'state': state, 'options': options}


        elif state.get("step") == "select_style":
            if message in ["기본형", "이미지형", "아이템리스트형"]:
                state["selected_style"] = message
                if message == "이미지형":
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
            # 사용자가 기존 템플릿을 선택했는지, 새로 만드는지 분기 처리
            if 'selected_template_content' in state:
                base_template = state['selected_template_content']
                # 이전에 선택한 템플릿이 이미지 포함인지 추정
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
            # ... (이하 로직은 이전과 동일)
            state['validation_result'] = validation_result
            state['correction_attempts'] = 0

            if validation_result['status'] == 'accepted':
                state['step'] = 'completed'
                return process_chat_message(message, state)
            else:
                state['step'] = 'correction'
                return {'message': f'템플릿을 생성했지만 규정 위반이 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result.get("suggestion", "없음")}\n\nAI가 자동으로 수정하겠습니다.', 'state': state}
                
        # ... (correction, manual_correction, completed 단계는 이전 답변과 동일하게 유지) ...
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
                    return process_chat_message(message, state)
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
                    return {'message': f'수정하신 템플릿에도 문제가 있습니다. 다시 수정해주세요.', 'state': state, 'options': ['포기하기']}


        elif state['step'] == 'completed':
            final_filled_template = state.get("template_draft", "")
            structured_data = structure_template_with_llm(final_filled_template)
            
            base_template = state.get("base_template", final_filled_template)
            variables = state.get('variables_info', [])
            editable_variables = {"parameterized_template": base_template, "variables": variables} if variables else None
            
            has_image_flag = state.get('hasImage', False)
            
            response_message = "✅ 템플릿이 생성되었습니다."

            state['step'] = 'initial' # 상태 초기화

            return {
                'message': response_message,
                'state': state,
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
    
    variables = re.findall(r'#\{(\w+)\}', template)
    
    if not variables:
        print("템플릿에 채울 변수가 없어 그대로 반환합니다.")
        return template

    variable_names = ", ".join([f"`#{v}`" for v in variables])

    prompt = ChatPromptTemplate.from_template(
        """당신은 주어진 템플릿과 사용자의 구체적인 요청을 결합하여 완성된 메시지를 만드는 전문가입니다.
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
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    
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
            "공통": """
        - GEN-PREVIEW-001 (미리보기 메시지 제한): 채팅방 리스트와 푸시에 노출되는 문구. 한/영 구분 없이 40자까지 입력 가능. 변수 작성 불가.
        - GEN-REVIEW-001 (심사 기본 원칙): 알림톡은 정보통신망법과 카카오 내부 기준에 따라 심사되며, 승인된 템플릿만 발송 가능.
        - GEN-REVIEW-002 (주요 반려 사유): 변수 오류, 과도한 변수(40개 초과) 사용, 변수로만 이루어진 템플릿, 변수가 포함된 버튼명, 변수가 포함된 미리보기 메시지 설정 시 반려됨.
        - GEN-INFO-DEF-001 (정보성 메시지의 정의): 고객의 요청에 의한 1회성 정보, 거래 확인, 계약 변경 안내 등이 포함됨. 부수적으로 광고가 포함되면 전체가 광고성 정보로 간주됨.
        - GEN-SERVICE-STD-001 (알림톡 서비스 기준): 알림톡은 수신자에게 반드시 전달되어야 하는 '정형화된 정보성' 메시지에 한함.
        - GEN-BLACKLIST-001 (블랙리스트 - 포인트/쿠폰): 수신자 동의 없는 포인트 적립/소멸 메시지, 유효기간이 매우 짧은 쿠폰 등은 발송 불가.
        - GEN-BLACKLIST-002 (블랙리스트 - 사용자 행동 기반): 장바구니 상품 안내, 클릭했던 상품 안내, 생일 축하 메시지, 앱 다운로드 유도 등은 발송 불가.
        - GEN-GUIDE-001 (정보성/광고성 판단 기준): 특가/할인 상품 안내, 프로모션 또는 이벤트가 혼재된 경우는 광고성 메시지로 판단됨.
        """,
            "기본형": {
                "규칙": """
        - GEN-TYPE-001 (기본형 특징 및 제한): 고객에게 반드시 전달되어야 하는 정보성 메시지. 한/영 구분 없이 1,000자까지 입력 가능하며, 개인화된 텍스트 영역은 #{변수}로 작성.
        - GEN-TYPE-002 (부가 정보형 특징 및 제한): 고정적인 부가 정보를 본문 하단에 안내. 최대 500자, 변수 사용 불가, URL 포함 가능. 본문과 합쳐 총 1,000자 초과 불가.
        - GEN-TYPE-003 (채널추가형 특징 및 제한): 비광고성 메시지 하단에 채널 추가 유도. 안내 멘트는 최대 80자, 변수/URL 포함 불가.
        """,
                "스타일 가이드": """
        # 스타일 설명: 텍스트 중심으로 정보를 전달하는 가장 기본적인 템플릿입니다. 간결하고 직관적인 구성으로 공지, 안내, 상태 변경 등 명확한 내용 전달에 사용됩니다.
        # 대표 예시 1 (서비스 완료 안내)
        안녕하세요, #{수신자명}님. 요청하신 #{서비스} 처리가 완료되었습니다. 자세한 내용은 아래 버튼을 통해 확인해주세요.
        # 대표 예시 2 (예약 리마인드)
        안녕하세요, #{수신자명}님. 내일(#{예약일시})에 예약하신 서비스가 예정되어 있습니다. 잊지 말고 방문해주세요.
        """
            },
            "이미지형": {
                "규칙": """
        - GEN-STYLE-001 (이미지형 특징 및 제한): 포맷화된 정보성 메시지를 시각적으로 안내. 광고성 내용 포함 불가. 템플릿 당 하나의 고정된 이미지만 사용 가능.
        - GEN-STYLE-002 (이미지형 제작 가이드 - 사이즈): 권장 사이즈는 800x400px (JPG, PNG), 최대 500KB.
        - GEN-STYLE-009 (이미지 저작권 및 내용 제한): 타인의 지적재산권, 초상권을 침해하는 이미지, 본문과 관련 없는 이미지, 광고성 이미지는 절대 사용 불가.
        """,
                "스타일 가이드": """
        # 스타일 설명: 시각적 요소를 활용하여 사용자의 시선을 끌고 정보를 효과적으로 전달하는 템플릿입니다. 상품 홍보, 이벤트 안내 등 시각적 임팩트가 중요할 때 사용됩니다.
        # 대표 예시 1 (신상품 출시)
        (이미지 영역: 새로 출시된 화장품 라인업 이미지)
        [신상품 출시] #{수신자명}님, 오래 기다리셨습니다! 새로운 #{브랜드명} 컬렉션이 출시되었습니다. 지금 바로 만나보세요.
        """
            },
            "아이템리스트형": {
                "규칙": """
        - GEN-STYLE-004 (아이템리스트형 특징 및 제한): 구조화된 정보(주문 내역 등) 안내에 사용. 광고성 내용 포함 불가. 아이템은 최소 2개 ~ 최대 10개까지 등록 가능.
        - GEN-STYLE-005 (아이템리스트형 제작 가이드 - 텍스트 길이): 아이템명은 최대 6자, 디스크립션은 최대 23자. 헤더는 최대 16자.
        - GEN-STYLE-008 (취소선 적용 가이드): 아이템 하이라이트/강조표기형의 타이틀 영역에 적용 가능. 텍스트 마지막에 `\\s` 입력 시 자동 변환.
        """,
                "스타일 가이드": """
        # 스타일 설명: 여러 항목을 목록 형태로 명확하게 나열하여 정보를 전달하는 템플릿입니다. 주문 내역, 단계별 안내, 핵심 요약 등 구조화된 정보 전달에 유용합니다.
        # 대표 예시 1 (주문 내역 안내)
        #{수신자명}님, 주문하신 내역입니다.
        - 주문번호: #{주문번호}
        - 주문상품: #{주문상품}
        - 총 결제금액: #{총금액}
        자세한 내용은 아래 버튼을 통해 확인해주세요.
        """
            }
        }

        if style == "기본형":
            internal_style_key = "기본형"
        elif style == "이미지형":
            internal_style_key = "이미지형"
        elif style == "아이템리스트형":
            internal_style_key = "아이템리스트형"
        else:
            internal_style_key = "기본형"

        RULE = RULES["공통"] + "\n" + RULES.get(internal_style_key, RULES["기본형"])["규칙"] + "\n" + RULES.get(internal_style_key, RULES["기본형"])["스타일 가이드"]

        # generation_rules = "" # 필요 시 RAG 로직 추가 가능

        generation_prompt = ChatPromptTemplate.from_template(
            """당신은 사용자의 요청을 분석하여 최적의 알림톡 텍스트 템플릿을 생성하는 AI 전문가입니다.
            # 최종 목표: 사용자의 요청과 규칙/가이드를 분석하여, 완성도 높은 템플릿 텍스트를 생성합니다.
            # 관련 규칙 및 가이드:
            {rules}
            # 사용자 요청:
            {request}
            # 지시사항:
            1. 최종 결과물은 제목, 본문, 버튼이 줄바꿈으로 구분된 텍스트여야 합니다.
            2. 규칙과 가이드의 스타일(어조, 구조)을 반드시 준수하세요.
            3. 바뀔 수 있는 정보는 `#{{변수명}}` 형식으로 만드세요.
            4. **이미지 포함 여부는 고려하지 마세요.** 오직 텍스트 내용에만 집중하세요.
            5. 다른 설명 없이, 최종 템플릿 텍스트만 출력하세요.
            """
        )
        
        generation_chain = generation_prompt | llm | StrOutputParser()
        
        template = generation_chain.invoke({
            "request": request,
            "rules": RULE,
        })
        
        return template.strip().strip('"`')
        
    except Exception as e:
        print(f"Error in generate_template: {e}")
        return "템플릿 생성 중 오류가 발생했습니다."

def validate_template(draft: str) -> dict:
    try:
        if 'compliance' not in retrievers or not retrievers['compliance']:
            print("🚨 경고: compliance 리트리버가 없어 검증을 건너뜁니다.")
            rules_with_metadata = "기본 규정: 광고성 문구, 욕설, 비방을 포함하지 마세요."
        else:
            compliance_docs = retrievers['compliance'].invoke(draft)
            rules_with_metadata = "\n".join([f"[규칙 ID: {doc.metadata.get('rule_id', 'unknown')}] {doc.page_content}" for doc in compliance_docs])

        if 'rejected' not in retrievers or not retrievers['rejected']:
            rejections = "기본 반려 사례: '파격 할인'과 같은 직접적인 광고 문구는 반려됩니다."
        else:
            rejected_docs = retrievers['rejected'].invoke(draft)
            rejections = "\n".join([doc.page_content for doc in rejected_docs])

        parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
        
        validation_prompt = ChatPromptTemplate.from_template(
            """당신은 알림톡 템플릿의 규정 준수 여부를 판단하는 전문가입니다.
            주어진 템플릿 초안을 검토하고, 규정 위반 사항이 있는지 분석해주세요.
            # 검토할 템플릿 초안:
            {draft}
            # 준수해야 할 규칙들:
            {rules}
            # 과거 반려된 템플릿 사례들:
            {rejections}
            # 지시사항:
            1. 템플릿이 규칙을 위반하는지 꼼꼼히 검토하세요.
            2. 위반 사항이 없다면 'status'를 'accepted'로 설정하세요.
            3. 위반 사항이 있다면 'status'를 'rejected'로 설정하고, 'suggestion'에 구체적인 개선 방안을 제시하세요.
            # 출력 형식 (JSON):
            {format_instructions}
            """
        )
        validation_chain = validation_prompt | llm | parser
        result = validation_chain.invoke({
            "draft": draft, 
            "rules": rules_with_metadata, 
            "rejections": rejections, 
            "format_instructions": parser.get_format_instructions()
        })
        if 'revised_template' in result:
            del result['revised_template']
        return result
    except Exception as e:
        print(f"Error in validate_template: {e}")
        return {
            "status": "accepted",
            "reason": "검증 중 오류가 발생했지만 템플릿을 승인합니다.",
            "evidence": None,
            "suggestion": None
        }

def correct_template(state: dict) -> str:
    try:
        attempts = state.get('correction_attempts', 0)

        if attempts == 0:
            instruction = """4. **(1차 수정)** '반려 사유'를 바탕으로 광고성 표현을 정보성 표현으로 순화하세요."""
        elif attempts == 1:
            instruction = """4. **(2차 수정: 금지어 적용)** '쿠폰', '할인', '이벤트', '특가' 등 직접적인 광고 용어 사용을 금지합니다."""
        else:
            instruction = """4. **(최종 수정: 관점 전환)** 메시지의 주체를 '사업자'에서 '고객'으로 전환하여 고객에게 유용한 정보를 안내하는 관점으로 수정하세요."""

        correction_prompt_template = """
        당신은 알림톡 템플릿 검수 전문가입니다. 광고성 메시지를 정보성 메시지로 수정하는 임무를 수행합니다.
        
        ### [분석 정보]
        1.  **최초 요청 의도:** {original_request}
        2.  **기존 템플릿 초안:** ```{rejected_draft}```
        3.  **반려 사유:** {rejection_reason}

        ### [지시사항]
        1.  '반려 사유'를 반드시 해결하세요.
        2.  '최초 요청 의도'는 유지해야 합니다.
        {dynamic_instruction}
        3.  최종 결과는 수정된 알림톡 템플릿 내용만 포함해야 합니다.

        ### [수정된 알림톡 템플릿]
        """
        correction_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
        correction_prompt = correction_prompt.partial(dynamic_instruction=instruction)
        
        correction_chain = correction_prompt | llm | StrOutputParser()

        rejection_info = state['validation_result']['reason']
        if state['validation_result'].get('suggestion'):
            rejection_info += "\n개선 제안: " + state['validation_result']['suggestion']

        new_draft = correction_chain.invoke({
            "original_request": state['original_request'],
            "rejected_draft": state['template_draft'],
            "rejection_reason": rejection_info
        })
        
        return new_draft.strip().strip('"`')

    except Exception as e:
        print(f"Error in correct_template: {e}")
        traceback.print_exc()
        return state.get('template_draft', '수정 중 오류가 발생했습니다.')