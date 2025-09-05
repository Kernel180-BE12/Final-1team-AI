import os
import json
import re
from typing import TypedDict, List, Optional, Dict
import sys
import traceback

# Pydantic 및 LangChain 호환성을 위한 임포트
from pydantic import BaseModel, Field, PrivateAttr

# LangChain 및 관련 라이브러리 임포트 (오류 수정됨)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# [오류 수정] 최신 LangChain 경로로 변경
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
        # [안정성 강화] LLM 출력이 비정상적일 경우를 대비
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
            
            return {
                'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다:\n\n' + '해당 템플릿중에서 사용하실 템플릿을 선택하시거나, 새로운 템플릿 생성을 원하시면 "새로 만들기"를 선택해주세요.',
                'state': state,
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
        traceback.print_exc()
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
        # [수정됨] style_guides 딕셔너리의 키를 함수 로직과 일치시킴
        style_guides = {
            "기본형_공식": """
            # 스타일 설명: 텍스트 중심으로 정보를 전달하는 가장 기본적인 템플릿입니다. 간결하고 직관적인 구성으로 공지, 안내, 상태 변경 등 명확한 내용 전달에 사용됩니다.
            # 대표 예시 1 (서비스 신청 완료):
            안녕하세요, #{수신자명}님. 요청하신 서비스 신청이 정상적으로 완료되었습니다. 자세한 내용은 아래 버튼을 통해 확인해주세요.
            # 대표 예시 2 (입금 확인):
            안녕하세요, #{수신자명}님. 주문하신 상품에 대한 입금이 확인되었습니다. 빠른 시일 내에 상품을 발송해드리겠습니다.
            # 대표 예시 3 (회원가입 환영):
            #{수신자명}님, 저희 회원이 되신 것을 진심으로 환영합니다. 앞으로 더 좋은 서비스로 보답하겠습니다.
            """,
            "이미지형_상품홍보": """
            # 스타일 설명: 시각적 요소를 활용하여 사용자의 시선을 끌고 정보를 효과적으로 전달하는 템플릿입니다. 상품 홍보, 이벤트 안내 등 시각적 임팩트가 중요할 때 사용됩니다.
            # 대표 예시 1 (신상품 출시):
            (이미지 영역: 새로 출시된 스니커즈 이미지)
            [신상품 출시] #{수신자명}님, 오래 기다리셨습니다! 드디어 새로운 스니커즈 컬렉션이 출시되었습니다. 지금 바로 만나보세요.
            # 대표 예시 2 (이벤트 참여 유도):
            (이미지 영역: 이벤트 관련 경품 이미지)
            [특별 이벤트] #{수신자명}님을 위한 특별 이벤트! 지금 참여하고 푸짐한 경품의 주인공이 되어보세요.
            """,
            "아이템리스트형_주문내역": """
            # 스타일 설명: 여러 항목을 목록 형태로 명확하게 나열하여 정보를 전달하는 템플릿입니다. 주문 내역, 단계별 안내, 핵심 요약 등 구조화된 정보 전달에 유용합니다.
            # 대표 예시 1 (주문 내역 안내):
            #{수신자명}님, 주문하신 상품 내역입니다.
            - 주문번호: #{주문번호}
            - 상품명: #{상품명} 외 2건
            - 총 결제금액: #{총금액}
            자세한 내용은 아래 버튼을 통해 확인해주세요.
            # 대표 예시 2 (단계별 안내):
            [이벤트 참여 방법 안내]
            1. 앱 최신 버전으로 업데이트하기
            2. 이벤트 페이지에서 '응모하기' 버튼 클릭
            3. 마케팅 수신 동의 확인하기
            지금 바로 참여하고 혜택을 받아보세요!
            """
        }

        # 사용자의 스타일 선택을 내부적으로 사용할 키 이름으로 변환
        if style == "기본형":
            internal_style_key = "기본형_공식"
        elif style == "이미지형":
            internal_style_key = "이미지형_상품홍보"
        elif style == "아이템리스트형":
            internal_style_key = "아이템리스트형_주문내역"
        else:
            # 예외 처리: 예상치 못한 스타일 값이 들어올 경우 기본값으로 설정
            internal_style_key = "기본형_공식"

        # [수정됨] get 메서드를 사용하여 안정적으로 스타일 가이드를 가져옴
        style_guide = style_guides.get(internal_style_key, style_guides["기본형_공식"])

        if 'generation' not in retrievers or not retrievers['generation']:
            print("🚨 경고: generation 리트리버가 없어 기본 프롬프트로 생성합니다.")
            generation_rules = "사용자의 요청에 맞춰 정보성 템플릿을 생성하세요."
        else:
            generation_docs = retrievers['generation'].invoke(request)
            generation_rules = "\n".join([doc.page_content for doc in generation_docs])
        
        generation_prompt = ChatPromptTemplate.from_template(
            """당신은 사용자의 요청을 분석하여 최적의 알림톡 템플릿을 생성하는 AI 전문가입니다.

# 최종 목표: 사용자의 요청과 선택된 스타일 가이드를 종합적으로 분석하여, 가장 적절하고 완성도 높은 템플릿 텍스트를 생성합니다.

# 스타일 가이드:
{style_guide}

# 생성 규칙 (부가 정보):
{generation_rules}

# 사용자의 구체적인 요청:
{request}

# 지시사항:
1. **스타일 심층 분석**: 주어진 '스타일 가이드'의 설명과 예시를 깊이 이해하고, 해당 스타일의 핵심적인 특징(어조, 구조, 형식)을 파악하여 생성물에 완벽하게 반영해야 합니다.
2. **창의적 재해석**: 사용자의 요청이 특정 스타일과 직접적으로 어울리지 않더라도, 요청의 본질적인 의도를 파악하고 이를 선택된 스타일에 맞게 창의적으로 재해석하여 템플릿을 구성해야 합니다. (예: '시험 일정 안내' 요청 + '이미지형' 스타일 -> 시험을 상징하는 달력이나 책 이미지를 가정하고, 응원 메시지를 담은 감성적인 문구로 재구성)
3. **변수화**: 고객명, 주문번호, 날짜, 금액 등 가변적인 정보는 #{{변수명}} 형식으로 반드시 변수 처리해야 합니다. 이를 통해 템플릿의 재사용성을 높입니다.
4. **구조적 완성도**: 최종 결과물은 [제목], [본문], [버튼]의 각 파트가 명확하게 줄바꿈(\n)으로 구분되어야 합니다. 단, 스타일의 특성에 따라 제목이나 버튼이 생략될 수 있습니다.
5. **특정 스타일 규칙 준수**:
    - **이미지형**: `(이미지 영역: ...)` 부분에 어떤 이미지가 들어가야 할지 구체적이고 명확하게 서술하여, 디자이너나 다른 AI가 이미지를 생성할 때 참고할 수 있도록 합니다.
    - **아이템리스트형**: #{{아이템리스트}}와 같은 변수 영역에는 사용자의 요청에 맞춰 `- 항목1: 내용1\n- 항목2: 내용2` 또는 `1. 내용1\n2. 내용2` 와 같이 명확한 목록 형식으로 내용을 채워야 합니다.
6. **결과물 형식**: 추가적인 설명이나 주석 없이, 오직 완성된 템플릿 텍스트만을 결과로 출력해야 합니다.
"""
        )
        
        generation_chain = generation_prompt | llm | StrOutputParser()
        
        # [수정됨] invoke에 전달하는 키 이름을 프롬프트의 변수명과 일치시킴 ('rules' -> 'generation_rules')
        template = generation_chain.invoke({
            "request": request,
            "style_guide": style_guide,
            "generation_rules": generation_rules
        })
        
        return template.strip().strip('"`')
        
    except Exception as e:
        print(f"Error in generate_template: {e}")
        # 오류 발생 시 사용자에게 전달될 기본 템플릿
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
    """
    [수정됨] 예시 기반 학습(Few-shot)을 포함한 최적화된 프롬프트를 사용하여
    알림톡 템플릿을 수정하는 함수.
    """
    try:
        attempts = state.get('correction_attempts', 0)

        # --- 동적 지시사항 설정 ---
        if attempts == 0:
            instruction = """4. **(1차 수정)** '반려 사유'를 바탕으로 광고성 표현을 정보성 표현으로 순화하세요. 학습한 '좋은 수정' 예시처럼 객관적인 정보 전달에 집중하세요."""
        elif attempts == 1:
            instruction = """4. **(2차 수정: 금지어 적용)** 여전히 광고성 문구가 남아있습니다. '쿠폰', '할인', '이벤트', '특가', '무료', '증정', '혜택'과 같이 직접적인 광고 및 마케팅 용어의 사용을 전면 금지합니다. 학습한 예시들처럼 이러한 단어 없이 정보 전달의 목적을 달성하세요."""
        else:
            instruction = """4. **(최종 수정: 관점 전환)** 이것이 마지막 기회입니다. 기존의 접근법을 완전히 버리세요.
            *   **관점 전환:** 메시지의 주체를 '우리(사업자)'에서 '고객님'으로 100% 전환하세요. (예: "저희가 준비했습니다" -> "고객님께서 받으실 수 있습니다")
            *   **목적 재정의:** '판매'나 '방문 유도'가 아닌, '고객님의 정보 수신 동의에 따라, 고객님의 권리(예: 보유 포인트, 회원 등급 혜택)에 대한 정보를 안내'하는 것으로 목적을 재설정하세요. 고객에게 유용한 정보를 제공하는 비서의 역할을 수행해야 합니다."""

        # --- [오류 수정] 예시 변수를 이중 중괄호로 감싼 프롬프트 템플릿 ---
        correction_prompt_template = """
당신은 대한민국 최고의 알림톡 템플릿 검수 전문가이자 카피라이터입니다.
당신의 임무는 정보통신망법과 KISA의 가이드라인을 철저히 준수하여, '광고성' 메시지를 '정보성' 메시지로 완벽하게 탈바꿈시키는 것입니다.

먼저 아래의 **[성공 및 실패 예시]**를 통해 좋은 템플릿과 나쁜 템플릿의 기준을 학습하세요.
그 다음, **[분석해야 할 정보]**를 바탕으로 **[따라야 할 지시사항]**에 맞춰 **[수정된 알림톡 템플릿]** 하나만을 생성해야 합니다.
그 외의 어떤 설명이나 인사말도 절대 포함해서는 안 됩니다.

### [분석해야 할 정보 (Analysis Target)]

1.  **최초 요청 의도 (The Goal):**
    {original_request}

2.  **기존 템플릿 초안 (Rejected Draft):**
    ```
    {rejected_draft}
    ```

3.  **반려 사유 및 개선 방향 (Rejection Analysis):**
    {rejection_reason}
---

### [따라야 할 지시사항 (Instructions)]

1.  **문제 해결:** '반려 사유 및 개선 방향'에 명시된 모든 문제점을 반드시 해결하세요. 이것이 최우선 과제입니다.
2.  **의도 유지:** '최초 요청 의도'에 담긴 핵심 목표(예: 예약 정보 전달, 배송 현황 안내)는 유지해야 합니다.
{dynamic_instruction}
4.  **엄격한 형식 준수:** 최종 결과물은 오직 알림톡 템플릿 내용만 포함해야 합니다. 따옴표나 코드 블록 없이, 순수 텍스트로만 응답하세요.

---
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
