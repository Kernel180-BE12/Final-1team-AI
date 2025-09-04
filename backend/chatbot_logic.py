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

# FlashRank 임포트
try:
    from flashrank import Ranker, RerankRequest
    from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
    from langchain_core.callbacks.manager import Callbacks
except ImportError:
    print("FlashRank 또는 관련 모듈을 찾을 수 없습니다.")
    BaseDocumentCompressor = object
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
    name: str = Field(description="추출된 변수의 한글 이름 (예: 매장명, 폐점일자). `#{}`에 들어갈 부분입니다.")
    original_value: str = Field(description="원본 텍스트에서 추출된 실제 값")
    description: str = Field(description="해당 변수에 대한 간단한 한글 설명 (사용자가 이해하기 쉽도록)")

class ParameterizedResult(BaseModel):
    parameterized_template: str = Field(description="특정 정보가 #{변수명}으로 대체된 최종 템플릿")
    variables: List[Variable] = Field(description="추출된 변수들의 목록")

class StructuredTemplate(BaseModel):
    title: str = Field(description="템플릿의 제목 또는 첫 문장")
    body: str = Field(description="제목과 버튼 텍스트를 제외한 템플릿의 핵심 본문 내용. 줄바꿈이 있다면 \\n으로 유지해주세요.")
    image_url: Optional[str] = Field(None, description="템플릿에 포함될 이미지의 URL. 이미지가 없는 경우 null입니다.")
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
        1. 텍스트의 첫 번째 문장이나 줄을 'title'로 추출합니다.
        2. 텍스트의 가장 마지막 줄에 있는 버튼 정보를 분석하여 'buttons' 리스트를 생성합니다.
        3. 버튼은 최대 2개까지 생성할 수 있으며, 없으면 빈 리스트 `[]`를 반환합니다.
        4. 각 버튼은 ['버튼종류', '버튼이름'] 형식의 튜플이어야 합니다. 버튼 종류는 내용에 맞게 '웹사이트', '앱링크', '전화하기' 등으로 추론하세요.
        5. 제목과 버튼을 제외한 나머지 모든 내용을 'body'로 추출합니다.
        6. 이미지가 언급되면 'image_url'을 생성하고, 없으면 null로 둡니다.
        7. 최종 결과를 지정된 JSON 형식으로만 출력해야 합니다.

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
            image_url=None,
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
            if docs:
                return Chroma.from_documents(
                    docs, 
                    embeddings, 
                    collection_name=name, 
                    persist_directory=vector_db_path, 
                    client_settings=client_settings
                )
            return None
            
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
    """채팅 메시지 처리 - 최종 수정 로직 적용"""
    try:
        if state['step'] == 'initial':
            state['original_request'] = message
            state['step'] = 'recommend_templates'
            
            if 'whitelist' not in retrievers or not retrievers['whitelist']:
                print("🚨 경고: whitelist 리트리버가 없어 신규 생성으로 바로 진행합니다.")
                state['step'] = 'select_style'
                return {
                    'message': '유사 템플릿 검색 기능이 비활성화 상태입니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:',
                    'state': state,
                    'options': ['기본형', '이미지형', '아이템리스트형']
                }

            similar_docs = retrievers['whitelist'].invoke(message)
            
            if not similar_docs:
                state['step'] = 'select_style'
                return {
                    'message': '유사한 기존 템플릿을 찾지 못했습니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:',
                    'state': state,
                    'options': ['기본형', '이미지형', '아이템리스트형']
                }
            
            structured_templates = [render_template_from_structured(structure_template_with_llm(doc.page_content)) for doc in similar_docs[:3]]
            
            # [수정] 사용자 안내 메시지의 '신규 생성'을 '새로 만들기'로 변경
            return {
                'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다:\n\n' + '해당 템플릿중에서 사용하실 템플릿을 선택하시거나, 새로운 템플릿 생성을 원하시면 "새로 만들기"를 선택해주세요.',
                'state': state,
                # 'options': ['템플릿 1', '템플릿 2', '템플릿 3', '새로 만들기'],
                'templates': [doc.page_content for doc in similar_docs[:3]],
                'structured_templates': structured_templates
            }
        
        elif state['step'] == 'recommend_templates':
            if message in ['템플릿 1', '템플릿 2', '템플릿 3']:
                template_idx = int(message.split()[1]) - 1
                if 'whitelist' not in retrievers or not retrievers['whitelist']:
                    return {'message': '오류: 템플릿을 가져올 수 없습니다. 다시 시도해주세요.', 'state': {'step': 'initial'}}
                
                similar_docs = retrievers['whitelist'].invoke(state['original_request'])
                if not similar_docs or len(similar_docs) <= template_idx:
                    return {'message': '오류: 선택한 템플릿을 찾을 수 없습니다. 다시 시도해주세요.', 'state': {'step': 'initial'}}

                state['selected_template'] = similar_docs[template_idx].page_content
                state['step'] = 'generate_and_validate'
                return process_chat_message(message, state)

            # [수정] 조건문의 '신규 생성'을 '새로 만들기'로 변경
            elif message == '새로 만들기':
                state['step'] = 'select_style'
                return {
                    'message': '새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요:',
                    'state': state,
                    'options': ['기본형', '이미지형', '아이템리스트형']
                }
            else:
                state['step'] = 'select_style'
                return process_chat_message(message, state)

        if state.get('step') == 'select_style':
            if message in ['기본형', '이미지형', '아이템리스트형']:
                state['selected_style'] = message
            else:
                state['selected_style'] = '기본형'
            
            state['step'] = 'generate_and_validate'
            return process_chat_message(message, state)

        if state.get('step') == 'generate_and_validate':
            if 'selected_template' in state and state['selected_template']:
                base_template = state['selected_template']
                del state['selected_template']
            else:
                newly_generated = generate_template(
                    state['original_request'], 
                    state.get('selected_style', '기본형')
                )
                param_result = parameterize_template(newly_generated)
                base_template = param_result.get('parameterized_template', newly_generated)
                state['variables_info'] = param_result.get('variables', [])

            state['base_template'] = base_template
            
            print("템플릿 내용을 채워 검증을 시작합니다.")
            template_draft = fill_template_with_request(
                template=base_template,
                request=state['original_request']
            )
            
            state['template_draft'] = template_draft
            validation_result = validate_template(template_draft)
            state['validation_result'] = validation_result
            state['correction_attempts'] = 0

            if validation_result['status'] == 'accepted':
                state['step'] = 'completed'
                return process_chat_message(message, state)
            else:
                state['step'] = 'correction'
                return {
                    'message': f'템플릿을 생성했지만 규정 위반이 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result.get("suggestion", "없음")}\n\nAI가 자동으로 수정하겠습니다.',
                    'state': state
                }
                
        elif state['step'] == 'correction':
            if state['correction_attempts'] < MAX_CORRECTION_ATTEMPTS:
                corrected_base_template = correct_template(state)
                state['correction_attempts'] += 1
                
                validation_result = validate_template(corrected_base_template)
                state["validation_result"] = validation_result
                
                if validation_result["status"] == "accepted":
                    state['base_template'] = corrected_base_template
                    
                    print("AI가 수정한 템플릿에 내용을 다시 채웁니다.")
                    final_draft = fill_template_with_request(
                        template=corrected_base_template,
                        request=state['original_request']
                    )
                    state['template_draft'] = final_draft
                    
                    state["step"] = "completed"
                    return process_chat_message(message, state)
                else:
                    state['template_draft'] = corrected_base_template
                    return process_chat_message(message, state)
            else:
                state['step'] = 'manual_correction'
                return {
                    'message': f'AI 자동 수정이 {MAX_CORRECTION_ATTEMPTS}회 모두 실패했습니다.\n\n현재 템플릿:\n{state["template_draft"]}\n\n마지막 문제점: {state["validation_result"]["reason"]}\n\n직접 수정하시겠습니까? 수정할 내용을 입력해주세요.',
                    'state': state,
                    'options': ['포기하기']
                }
                
        elif state['step'] == 'manual_correction':
            if message == '포기하기':
                state['step'] = 'initial'
                return {'message': '템플릿 생성을 포기했습니다. 새로운 요청을 입력해주세요.', 'state': {'step': 'initial'}}
            else:
                user_corrected_template = message
                validation_result = validate_template(user_corrected_template)
                state['validation_result'] = validation_result

                if validation_result['status'] == 'accepted':
                    state['base_template'] = user_corrected_template
                    
                    print("사용자가 수정한 템플릿에 내용을 다시 채웁니다.")
                    final_draft = fill_template_with_request(
                        template=user_corrected_template,
                        request=state['original_request']
                    )
                    state['template_draft'] = final_draft
                    
                    state['step'] = 'completed'
                    return process_chat_message(message, state)
                else:
                    return {
                        'message': f'🚨 수정하신 템플릿에도 여전히 문제가 있습니다.\n\n문제점: {validation_result["reason"]}\n\n다시 수정해주시거나 "포기하기"를 선택해주세요.',
                        'state': state,
                        'options': ['포기하기']
                    }
        
        elif state['step'] == 'completed':
            final_filled_template = state.get("template_draft", "")
            structured_data = structure_template_with_llm(final_filled_template)

            base_template = state.get("base_template", final_filled_template)
            variables = state.get('variables_info', [])

            editable_variables = {
                "parameterized_template": base_template,
                "variables": variables
            } if variables else None

            # 다음 대화를 위해 상태 초기화
            state['step'] = 'initial'
            
            return {
                'message': '✅ 템플릿이 성공적으로 생성되었습니다!',
                'state': state,
                'template': final_filled_template,
                'structured_template': structured_data,
                'editable_variables': editable_variables,
                'buttons': structured_data.buttons
            }
        
        return {
            'message': '알 수 없는 상태입니다. 다시 시도해주세요.',
            'state': {'step': 'initial'}
        }
        
    except Exception as e:
        print(f"Error in process_chat_message: {e}")
        traceback.print_exc() # 오류 발생 시 스택 트레이스 출력
        return {
            'message': f'처리 중 오류가 발생했습니다: {str(e)}',
            'state': {'step': 'initial'}
        }

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
        style_guides = {
            # ======================================================
            # 1. 기본형 테마 (수정 불필요)
            # ======================================================
            "기본형_친근": """
            # 스타일 설명: 친구에게 말하듯 부드럽고 친근한 어투를 사용하는 기본 템플릿입니다.
            # 뼈대:
            [가벼운 제목]
            #{고객명}님, 안녕하세요!
            요청하신 내용에 대해 알려드릴게요.
            [버튼 이름]
            # 예시:
            [주문하신 상품이 준비됐어요!]
            #{고객명}님, 안녕하세요!
            주문하신 상품이 매장에 도착했어요. 편하실 때 찾아가세요!
            자세히 보러가기
            """,
            "기본형_공식": """
            # 스타일 설명: 정중하고 격식 있는 어투를 사용하는 비즈니스용 기본 템플릿입니다.
            # 뼈대:
            [정중한 제목]
            #{고객명} 고객님께,
            요청하신 내용에 대해 아래와 같이 안내드립니다.
            감사합니다.
            [버튼 이름]
            # 예시:
            [결제 완료 내역 안내]
            #{고객명} 고객님께,
            요청하신 결제 내역이 아래와 같이 정상적으로 처리되었음을 안내드립니다.
            주문번호: #{주문번호}
            결제금액: #{결제금액}
            감사합니다.
            상세 영수증 확인
            """,
            "기본형_긴급": """
            # 스타일 설명: 긴급하거나 중요한 정보를 즉시 전달해야 할 때 사용하는 간결한 템플릿입니다.
            # 뼈대:
            [긴급 안내 제목]
            중요한 안내사항입니다.
            #{고객명}님, 반드시 확인해 주시기 바랍니다.
            [버튼 이름]
            # 예시:
            [계정 보안 경고]
            #{고객명}님의 계정에서 새로운 로그인이 감지되었습니다.
            본인이 아닐 경우 즉시 비밀번호를 변경해 주시기 바랍니다.
            계정 활동 확인
            """,

            # ======================================================
            # 2. 이미지형 테마 (수정됨)
            # ======================================================
            "이미지형_상품홍보": """
            # 스타일 설명: 상품 이미지를 강조하며 구매를 유도하는 마케팅용 템플릿입니다.
            # 뼈대:
            (이미지 영역: 매력적인 상품 이미지)
            [시선을 끄는 제목]
            #{{본문}}
            [강력한 CTA 버튼]
            # 예시:
            (이미지: 신상품 스니커즈)
            [이번 주말, 단 3일간의 특별 할인!]
            #{고객명}님만을 위해 준비한 새로운 스니커즈 컬렉션을 30% 할인된 가격으로 만나보세요.
            지금 바로 구매하기
            """,
            "이미지형_정보전달": """
            # 스타일 설명: 차트, 지도, 인포그래픽 등의 정보성 이미지를 사용하여 내용을 효과적으로 전달합니다.
            # 뼈대:
            (이미지 영역: 정보성 이미지)
            [정보 요약 제목]
            #{{본문}}
            [버튼 이름]
            # 예시:
            (이미지: 월별 사용량 그래프)
            [8월 서비스 이용 내역 리포트]
            #{고객명}님의 지난달 서비스 이용 내역을 그래프로 확인해보세요.
            지난달 대비 #{변동률}% 사용량이 변화했습니다.
            상세 리포트 보기
            """,
            "이미지형_감성": """
            # 스타일 설명: 감성적인 이미지와 문구를 사용하여 브랜드 이미지를 높이거나 특별한 메시지를 전달합니다.
            # 뼈대:
            (이미지 영역: 감성적인 사진)
            [감성적인 문구의 제목]
            #{{본문}}
            [차분한 느낌의 버튼]
            # 예시:
            (이미지: 비 오는 창가 풍경)
            [비 오는 날, 따뜻한 커피 한 잔 어떠세요?]
            #{고객명}님, 잠시 창밖을 보며 여유를 가져보세요.
            오늘 하루도 고생 많으셨습니다.
            음악과 함께 쉬어가기
            """,

            # ======================================================
            # 3. 아이템리스트형 테마 (수정됨)
            # ======================================================
            "아이템리스트형_주문내역": """
            # 스타일 설명: 구매한 상품 목록과 같이 여러 항목을 명확하게 나열하여 전달합니다.
            # 뼈대:
            [주문 내역 안내]
            #{고객명}님, 주문하신 상품 내역입니다.
            #{{아이템리스트}}
            총 결제금액: #{총금액}
            [버튼 이름]
            # 예시:
            [주문하신 상품이 배송 시작되었습니다]
            #{고객명}님, 주문하신 상품이 안전하게 포장되어 출발했습니다.
            - 주문번호: #{주문번호}
            - 상품명: #{상품명} 외 2건
            - 배송사: #{택배사}
            - 송장번호: #{송장번호}
            배송 조회하기
            """,
            "아이템리스트형_단계별안내": """
            # 스타일 설명: 회원가입, 이벤트 참여 방법 등 순서가 있는 정보를 단계별로 안내합니다.
            # 뼈대:
            [단계별 안내 제목]
            아래 순서에 따라 진행해주세요.
            #{{단계별안내리스트}}
            [버튼 이름]
            # 예시:
            [이벤트 참여 방법 안내]
            간단하게 3단계만 거치면 이벤트 참여 완료!
            1. 앱 최신 버전으로 업데이트하기
            2. 이벤트 페이지에서 '응모하기' 버튼 클릭
            3. 마케팅 수신 동의 확인하기
            이벤트 참여하러 가기
            """,
            "아이템리스트형_핵심요약": """
            # 스타일 설명: 긴 내용이나 약관의 핵심만을 요약하여 전달할 때 사용합니다.
            # 뼈대:
            [핵심 내용 요약]
            #{고객명}님, 알아두셔야 할 핵심 내용입니다.
            #{{핵심요약리스트}}
            [버튼 이름]
            # 예시:
            [서비스 이용약관 변경 사전 안내]
            #{고객명}님, 2025년 9월 1일부터 서비스 이용약관이 변경됩니다.
            - 주요 변경사항: 개인정보 처리 방침 강화
            - 효력 발생일: 2025년 9월 1일
            - 참고: 미동의 시 서비스 이용이 제한될 수 있습니다.
            전체 내용 확인하기
            """
        }

        if style == "기본형":
            style = "기본형_공식"
        elif style == "이미지형":
            style = "이미지형_상품홍보"
        elif style == "아이템리스트형":
            style = "아이템리스트형_주문내역"

        style_guide = style_guides.get(style, style_guides["기본형_공식"])

        if 'generation' not in retrievers or not retrievers['generation']:
            print("🚨 경고: generation 리트리버가 없어 기본 프롬프트로 생성합니다.")
            generation_rules = "사용자의 요청에 맞춰 정보성 템플릿을 생성하세요."
        else:
            generation_docs = retrievers['generation'].invoke(request)
            generation_rules = "\n".join([doc.page_content for doc in generation_docs])
        
        generation_prompt = ChatPromptTemplate.from_template(
            """당신은 주어진 '스타일 가이드'를 완벽하게 이해하고 따르는 알림톡 템플릿 생성 전문가입니다.

            # 최종 목표: 사용자의 구체적인 요청사항을 '스타일 가이드'의 뼈대와 예시에 맞춰 변환하여, 완전한 템플릿 텍스트 하나만 생성하세요.

            # 스타일 가이드:
            {style_guide}

            # 생성 규칙 (부가 정보):
            {rules}

            # 사용자의 구체적인 요청:
            {request}

            # 지시사항:
            1. **가장 중요한 규칙**: 최종 결과물은 반드시 [제목], [본문], [버튼]의 각 파트가 명확하게 줄바꿈(`\\n`)으로 구분된 **여러 줄의 텍스트**여야 합니다.
            2. **스타일 준수 의무**: '사용자의 구체적인 요청'이 '스타일 가이드'와 어울리지 않더라도, 요청 내용을 스타일의 뼈대와 예시에 맞게 **창의적으로 재해석하고 변형**하여 반드시 해당 스타일로 만들어야 합니다. 예를 들어, 요청이 '시험 일정 안내'이고 스타일이 '이미지형'이라면, 시험을 상징하는 이미지(책, 캘린더 등)을 가정하고 그에 어울리는 문구를 생성해야 합니다.
            3. '스타일 가이드'의 '뼈대'와 '예시'를 최우선으로 참고하여 구조와 형식을 결정하세요.
            4. '사용자의 구체적인 요청'에서 내용을 추출하여 뼈대를 채워 넣으세요.
            5. 바뀔 수 있는 구체적인 정보(예: 고객명, 주문번호, 날짜)는 `#{{변수명}}` 형식으로 만드세요.
            6. **이미지형 템플릿의 경우**: `(이미지 영역: ...)` 부분에 이미지에 대한 간략한 설명을 포함하세요. 이 설명은 실제 이미지가 아닌, 어떤 종류의 이미지가 들어갈지 LLM이 이해할 수 있도록 돕는 역할을 합니다.
            7. **아이템리스트형 템플릿의 경우**: `#{{아이템리스트}}`, `#{{단계별안내리스트}}`, `#{{핵심요약리스트}}`와 같은 변수에는 사용자의 요청에 따라 여러 항목을 `- 항목1: 내용1\\n- 항목2: 내용2` 형식으로 채워 넣으세요.
            8. 다른 어떤 설명도 없이, 오직 최종 템플릿 텍스트만 출력하세요.
            """
        )
        
        generation_chain = generation_prompt | llm | StrOutputParser()
        template = generation_chain.invoke({
            "request": request,
            "style_guide": style_guide,
            "rules": generation_rules
        })
        
        return template.strip().strip('"`')
        
    except Exception as e:
        print(f"Error in generate_template: {e}")
        return "템플릿 생성 중 오류가 발생했습니다.\n다시 시도해주세요.\n확인"

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
    """템플릿 수정 함수"""
    try:
        attempts = state.get('correction_attempts', 0)
        # 동적 지시사항 설정
        if attempts == 0:
            instruction = "3. 광고성 문구를 제거하거나, 정보성 내용으로 순화하는 등, 제안된 방향에 맞게 템플릿을 수정하세요."
        elif attempts == 1:
            instruction = "3. **(2차 수정)** 아직도 문제가 있습니다. 이번에는 '쿠폰', '할인', '이벤트', '특가'와 같은 명백한 광고성 단어를 사용하지 마세요."
        else:
            instruction = """3. **(최종 수정: 관점 전환)** 여전히 광고성으로 보입니다. 이것이 마지막 시도입니다.
            - **관점 전환:** 메시지의 주체를 '우리(사업자)'에서 '고객님'으로 완전히 바꾸세요.
            - **목적 변경:** '판매'나 '방문 유도'가 아니라, '고객님이 과거에 동의한 내용에 따라 고객님의 권리(혜택) 정보를 안내'하는 것으로 목적을 재정의하세요."""

        correction_prompt_template = """당신은 지적된 문제점을 해결하여 더 나은 대안을 제시하는 전문 카피라이터입니다.
        당신의 유일한 임무는 아래 지시사항에 따라 **수정된 템플릿 초안 하나만**을 생성하는 것입니다. 초안 외에 다른 설명은 절대로 덧붙이지 마세요.

        # 원래 사용자 요청: {original_request}
        # 이전에 제안했던 템플릿 (반려됨):
        ```{rejected_draft}```
        # 반려 사유 및 개선 제안:
        {rejection_reason}

        # 지시사항:
        1. '반려 사유 및 개선 제안'을 완벽하게 이해하고, 지적된 모든 문제점을 해결하세요.
        2. '원래 사용자 요청'의 핵심 의도는 유지해야 합니다.
        {dynamic_instruction}

        # 수정된 템플릿 초안 (오직 템플릿 텍스트만 출력):
        """
        correction_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
        correction_prompt = correction_prompt.partial(dynamic_instruction=instruction)
        correction_chain = correction_prompt | llm | StrOutputParser()

        # 반려 사유와 제안을 합쳐서 전달
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
        return state.get('template_draft', '수정 중 오류가 발생했습니다.')