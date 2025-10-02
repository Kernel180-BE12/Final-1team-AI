# import os
# import json
# import re
# import asyncio
# from typing import TypedDict, List, Optional, Dict, Any, Literal, Union
# import sys
# import traceback

# # Pydantic 및 LangChain 호환성을 위한 임포트
# from pydantic import BaseModel, Field, PrivateAttr

# # LangChain 및 관련 라이브러리 임포트
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.document_loaders.base import BaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain.retrievers import EnsembleRetriever
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain_community.retrievers import BM25Retriever
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from langchain.output_parsers.pydantic import PydanticOutputParser
# from dotenv import load_dotenv
# from langchain_core.documents.compressor import BaseDocumentCompressor
# from langchain_core.callbacks import Callbacks

# # FlashRank 임포트
# try:
#     from flashrank import Ranker, RerankRequest
# except ImportError:
#     print("FlashRank 또는 관련 모듈을 찾을 수 없습니다. Reranking 기능이 비활성화됩니다.")
#     Ranker = None

# # .env 파일에서 API 키 로드
# load_dotenv()

# # --- 설정 및 모델 정의 ---
# MAX_CORRECTION_ATTEMPTS = 3

# # --- 전역 변수 및 헬퍼 함수 ---
# llm_reasoning = None # gpt-5 (고성능 추론용)
# llm_fast = None      # gpt-4.1 (단순 작업용)
# llm_medium = None    # gpt-4.1-mini (단순 작업용)
# llm_general = None   # gpt-4o (새 코드에서 사용)
# retrievers = {}
# approved_templates = []
# rejected_templates = []

# class CustomRuleLoader(BaseLoader):
#     def __init__(self, file_path: str, encoding: str = 'utf-8'):
#         self.file_path = file_path
#         self.encoding = encoding
    
#     def load(self) -> List[Document]:
#         docs = []
#         try:
#             with open(self.file_path, 'r', encoding=self.encoding) as f:
#                 content = f.read()
#         except FileNotFoundError:
#             print(f"🚨 경고: '{self.file_path}' 파일을 찾을 수 없습니다.")
#             return []
        
#         rule_blocks = re.findall(r'\[규칙 시작\](.*?)\[규칙 끝\]', content, re.DOTALL)
#         for block in rule_blocks:
#             lines = block.strip().split('\n')
#             metadata = {}
#             page_content = ""
#             is_content_section = False
            
#             for line in lines:
#                 if line.lower().startswith('content:'):
#                     is_content_section = True
#                     page_content += line[len('content:'):].strip() + "\n"
#                     continue
#                 if is_content_section:
#                     page_content += line.strip() + "\n"
#                 else:
#                     if ':' in line:
#                         key, value = line.split(':', 1)
#                         metadata[key.strip()] = value.strip()
            
#             if page_content:
#                 docs.append(Document(page_content=page_content.strip(), metadata=metadata))
        
#         return docs

# class TemplateAnalysisResult(BaseModel):
#     status: str = Field(description="템플릿의 최종 상태")
#     reason: str = Field(description="상세한 판단 이유")
#     evidence: Optional[str] = Field(None, description="판단 근거 규칙들의 rule_id")
#     suggestion: Optional[str] = Field(None, description="개선 제안")

# class FlashRankRerank(BaseDocumentCompressor):
#     _ranker: Ranker = PrivateAttr()
#     top_n: int = 3
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if Ranker:
#             self._ranker = Ranker()
    
#     class Config:
#         arbitrary_types_allowed = True
    
#     def compress_documents(self, documents: List[Document], query: str, callbacks: Callbacks | None = None) -> List[Document]:
#         if not documents or not Ranker:
#             return documents[:self.top_n]
        
#         rerank_request = RerankRequest(
#             query=query,
#             passages=[{"id": i, "text": doc.page_content, "metadata": doc.metadata} for i, doc in enumerate(documents)]
#         )
#         reranked_results = self._ranker.rerank(rerank_request)
        
#         final_docs = []
#         for item in reranked_results[:self.top_n]:
#             doc = documents[item['id']]
#             doc.metadata["relevance_score"] = item['score']
#             final_docs.append(doc)
        
#         return final_docs

# class Variable(BaseModel):
#     name: str = Field(description="추출된 변수의 한글 이름 (예: 매장명, 폐점일자). `#{{}}`에 들어갈 부분입니다.")
#     original_value: str = Field(description="원본 텍스트에서 추출된 실제 값")
#     description: str = Field(description="해당 변수에 대한 간단한 한글 설명 (사용자가 이해하기 쉽도록)")

# class ParameterizedResult(BaseModel):
#     parameterized_template: str = Field(description="특정 정보가 #{{변수명}}으로 대체된 최종 템플릿")
#     variables: List[Variable] = Field(description="추출된 변수들의 목록")

# class StructuredTemplate(BaseModel):
#     title: str = Field(description="템플릿의 제목 또는 첫 문장")
#     body: str = Field(description="제목과 버튼 텍스트를 제외한 템플릿의 핵심 본문 내용. 줄바꿈이 있다면 \\n으로 유지해주세요.")
#     buttons: Optional[List[tuple[str, str]]] = Field(None, description="템플릿에 포함될 버튼 리스트. 예: [('웹사이트', '자세히 보기')]")

# class Button(BaseModel):
#     """템플릿 하단 버튼 모델"""
#     type: Literal["AL", "WL", "AC", "BK", "MD"] = Field(description="AL: 앱링크, WL: 웹링크, AC: 채널추가, BK: 봇키워드, MD: 메시지전달")
#     name: str = Field(max_length=28, description="버튼 이름 (최대 28자)")
#     value: str = Field(description="URL, 앱 경로, 봇키워드 등")

# class Highlight(BaseModel):
#     """강조표기형 타이틀/서브타이틀 모델"""
#     title: str = Field(max_length=23, description="강조형 타이틀 (최대 23자)")
#     subtitle: str = Field(max_length=18, description="강조형 서브타이틀 (변수 사용 불가, 최대 18자)")

# class Item(BaseModel):
#     """아이템 리스트 개별 항목 모델"""
#     name: str = Field(max_length=6, description="아이템명 (변수 사용 불가, 최대 6자)")
#     description: str = Field(max_length=23, description="설명 (변수 사용 가능, 최대 23자)")
#     summary: Optional[str] = Field(None, description="우측 요약 정보 (숫자, 통화기호 등만 가능)")

# class ItemHighlight(BaseModel):
#     """아이템 리스트 하이라이트 모델"""
#     thumbnail_required: bool = Field(default=True, description="썸일 필요 여부")
#     text: str = Field(max_length=21, description="하이라이트 텍스트 (최대 21자)")
#     description: str = Field(max_length=13, description="하이라이트 설명 (최대 13자)")

# class BasicTemplate(BaseModel):
#     """기본형 템플릿 모델"""
#     body: str = Field(max_length=1300, description="본문 (최대 1300자)")
#     footer: Optional[str] = Field(None, max_length=500, description="부가 정보 (최대 500자, 변수 불가)")
#     add_channel: Optional[bool] = Field(False, description="채널 추가 버튼 여부")
#     buttons: List[Button] = Field([], max_items=5, description="버튼 리스트 (최대 5개)")

# class ImageTemplate(BasicTemplate):
#     """이미지형 템플릿 모델 (기본형 상속)"""
#     image_url: str = Field(description="이미지 URL (800x400px 권장)")

# class HighlightTemplate(BasicTemplate):
#     """강조표기형 템플릿 모델 (기본형 상속)"""
#     highlight: Highlight = Field(description="강조형 타이틀/서브타이틀")

# class ItemListTemplate(BaseModel):
#     """아이템 리스트형 템플릿 모델"""
#     header: Optional[str] = Field(None, max_length=16, description="헤더 (최대 16자)")
#     item_highlight: Optional[ItemHighlight] = Field(None)
#     body: str = Field(max_length=1300)
#     items: List[Item] = Field(min_items=2, max_items=10)
#     buttons: List[Button] = Field([], max_items=5)

# class CompositeTemplate(BasicTemplate):
#     """복합형 템플릿 모델 (기본형 상속)"""
#     footer: str = Field(..., max_length=500, description="부가 정보 (필수)")
#     add_channel: bool = Field(True, description="채널 추가 (필수)")

# class TemplateResponse(BaseModel):
#     """최종 생성 결과 응답 모델"""
#     style: Literal["기본형", "이미지형", "강조형", "아이템 리스트형", "복합형"]
#     template_data: Union[BasicTemplate, ImageTemplate, HighlightTemplate, ItemListTemplate, CompositeTemplate]

# class ValidationResult(BaseModel):
#     """내부 검증 결과 모델"""
#     status: Literal["accepted", "rejected"]
#     reason: Optional[str] = None


# def load_line_by_line(file_path: str) -> List[str]:
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             items = [line.strip() for line in f if line.strip()]
#         return items
#     except FileNotFoundError:
#         return []

# def load_by_separator(file_path: str, separator: str = '---') -> List[str]:
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#         items = [section.strip() for section in content.split(separator) if section.strip()]
#         return items
#     except FileNotFoundError:
#         return []


# async def _create_and_run_chain(request: str, pydantic_model: BaseModel, system_prompt: str):
#     """LLM 체인을 생성하고 비동기로 실행하는 헬퍼 함수"""
#     parser = PydanticOutputParser(pydantic_object=pydantic_model)
#     prompt = ChatPromptTemplate(
#         messages=[
#             SystemMessagePromptTemplate.from_template(system_prompt),
#             HumanMessagePromptTemplate.from_template("사용자 요청: {request}"),
#         ],
#         input_variables=["request"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )
#     chain = prompt | llm_general | parser
#     return await chain.ainvoke({"request": request})

# async def generate_template_for_style(request: str, style: str) -> BaseModel:
#     """스타일에 맞는 생성기를 호출하여 Pydantic 객체를 생성"""
#     style_configs = {
#         "기본형": (BasicTemplate, "텍스트 중심의 표준 템플릿. 본문, 푸터, 버튼으로 구성."),
#         "이미지형": (ImageTemplate, "상단에 이미지가 포함된 템플릿. `image_url`은 적절한 예시 URL로 채워야 함."),
#         "강조형": (HighlightTemplate, "핵심 정보를 `highlight` 객체로 추출하여 주목도를 높이는 템플릿."),
#         "아이템 리스트형": (ItemListTemplate, "주문/예약 내역 등 반복 정보를 목록 형태로 보여주는 템플릿. `header`, `item_highlight`, `items` 등 모든 요소를 충실히 생성."),
#         "복합형": (CompositeTemplate, "`footer`와 `add_channel`이 필수로 포함되는 템플릿.")
#     }
    
#     if style not in style_configs:
#         raise ValueError(f"지원하지 않는 스타일입니다: {style}")

#     pydantic_model, style_desc = style_configs[style]
    
#     system_prompt = f"""당신은 카카오 알림톡 '{style}' 템플릿 제작 전문가입니다.
# 사용자 요청을 분석하여 '{style}' 스타일에 맞는 완벽한 JSON 객체를 생성하세요.
# {style_desc}
# 변수가 필요한 곳은 '#{{{{변수명}}}}' 형식을 사용하고, 모든 필드는 카카오 가이드라인 제약 조건(글자 수 등)을 엄격히 준수해야 합니다.

# # 출력 형식
# {{format_instructions}}"""
#     return await _create_and_run_chain(request, pydantic_model, system_prompt)


# async def validate_template_new(template_json: str) -> ValidationResult:
#     """생성된 템플릿을 검증하는 로직 (새 코드의 validate_template)"""
#     # 실제 구현 시: RAG로 규정 조회 후 LLM으로 위반 여부 판단
#     print("🤖 템플릿 내부 검증 수행...")
#     # 여기서는 항상 accepted를 반환하지만, 실제 구현에서는 LLM을 통해 검증 로직을 추가해야 합니다.
#     return ValidationResult(status="accepted")

# async def refine_template_with_feedback_new(state: dict, feedback: str) -> dict:
#     """피드백과 전체 대화 맥락을 반영하여 템플릿(Pydantic 객체)을 수정합니다."""
    
#     # 1. state에서 맥락 정보 추출
#     initial_request = state.get("original_request", "알 수 없음")
#     current_response = TemplateResponse(**state["final_template_response"])
#     original_template_obj = current_response.template_data
    
#     # Pydantic 객체를 딕셔너리와 JSON 문자열로 변환
#     template_dict = original_template_obj.model_dump()
#     template_json_str = json.dumps(template_dict, ensure_ascii=False, indent=2)
    
#     # AI가 참고할 수 있도록 현재 템플릿의 본문을 텍스트로 추출
#     current_template_text = template_dict.get("body", "본문 없음")

#     # 2. Pydantic 파서와 개선된 프롬프트 정의
#     parser = PydanticOutputParser(pydantic_object=type(original_template_obj))
    
#     system_prompt = """당신은 사용자의 피드백을 반영하여 JSON 형식의 템플릿을 수정하는 AI 전문가입니다.

# ### 맥락 정보
# - **사용자의 최초 요청 의도:** {initial_request}
# - **수정 전 템플릿 원본 텍스트:** ```
#   {current_template_text}
# 사용자의 수정 요청사항: {feedback}

# 작업 지시
# '맥락 정보'를 종합적으로 이해하여 사용자의 수정 의도를 명확히 파악하세요.

# 사용자의 요청이 "이 부분", "저기" 등 모호하더라도, '수정 전 템플릿 원본 텍스트'를 참고하여 수정 대상을 정확히 추론해야 합니다.

# 사용자가 '제거' 또는 '삭제'를 요청하면, 다른 지시보다 우선하여 해당 내용을 반드시 제거해야 합니다.

# '최초 요청 의도'를 잃지 않으면서, 위 지시사항을 바탕으로 아래 '수정 대상 JSON'을 수정하세요.

# 최종 결과는 반드시 수정된 JSON 객체만 출력해야 합니다. 서론이나 부가 설명은 절대 포함하지 마세요.

# 출력 형식
# {format_instructions}
# """

#     human_prompt = """### 수정 대상 JSON
# JSON

# {template_json}
# 사용자의 수정 요청사항
# {feedback}

# 수정된 템플릿 (JSON):
# """

#     prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", human_prompt)
# ])

# # 3. 체인 생성 및 실행
#     chain = prompt | llm_general | parser

#     refined_obj = await chain.ainvoke({ 
#     "initial_request": initial_request,
#     "current_template_text": current_template_text,
#     "template_json": template_json_str, 
#     "feedback": feedback,
#     "format_instructions": parser.get_format_instructions()
# })

#     return refined_obj.model_dump()

# def initialize_system():
#     global llm_reasoning, llm_fast, llm_medium, llm_general, retrievers, approved_templates, rejected_templates
#     if llm_reasoning is not None:
#         return

#     print("서버 시작: 시스템 초기화를 진행합니다...")
#     try:
#         data_dir = 'data'
    
#         llm_reasoning = ChatOpenAI(model="gpt-4.1", temperature=0.3)
#         llm_medium = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
#         llm_fast = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
#         llm_general = ChatOpenAI(model="gpt-4o", temperature=0.1) # 새 코드에서 사용

    
#         embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
#         approved_templates = load_line_by_line(os.path.join(data_dir, "approved_templates.txt"))
#         rejected_templates = load_by_separator(os.path.join(data_dir, "rejected_templates.txt"))
    
#         docs_compliance = CustomRuleLoader(os.path.join(data_dir, "compliance_rules.txt")).load()
#         docs_generation = CustomRuleLoader(os.path.join(data_dir, "generation_rules.txt")).load()
#         docs_whitelist = [Document(page_content=t) for t in approved_templates]
#         docs_rejected = [Document(page_content=t) for t in rejected_templates]
    
#         def create_db(name, docs):
#             if not docs:
#                 print(f"🚨 '{name}'에 대한 문서가 없어 DB 생성을 건너뜁니다.")
#                 return None
        
#             print(f"✨ '{name}' 컬렉션을 인메모리 DB에 생성합니다...")
#             db = Chroma.from_documents(
#                 docs, 
#                 embeddings, 
#                 collection_name=name
#             )
#             print(f"💾 '{name}' 컬렉션이 인메모리 DB에 저장되었습니다.")
#             return db

#         db_compliance = create_db("compliance_rules", docs_compliance)
#         db_generation = create_db("generation_rules", docs_generation)
#         db_whitelist = create_db("whitelist_templates", docs_whitelist)
#         db_rejected = create_db("rejected_templates", docs_rejected)
    
#         def create_hybrid_retriever(vectorstore, docs):
#             if not vectorstore:
#                 return None
#             vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#             if docs:
#                 keyword_retriever = BM25Retriever.from_documents(docs)
#                 keyword_retriever.k = 5
#                 ensemble_retriever = EnsembleRetriever(
#                     retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5]
#                 )
#             else:
#                 ensemble_retriever = vector_retriever
#             if Ranker:
#                 compressor = FlashRankRerank(top_n=5)
#                 return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
#             return ensemble_retriever

#         retrievers['compliance'] = create_hybrid_retriever(db_compliance, docs_compliance)
#         retrievers['generation'] = create_hybrid_retriever(db_generation, docs_generation)
#         retrievers['whitelist'] = create_hybrid_retriever(db_whitelist, docs_whitelist)
#         retrievers['rejected'] = create_hybrid_retriever(db_rejected, docs_rejected)

#         for name, retriever in retrievers.items():
#             if retriever:
#                 print(f"✅ '{name}' 리트리버가 성공적으로 생성되었습니다.")
#             else:
#                 print(f"🚨 경고: '{name}' 리트리버를 생성하지 못했습니다 (관련 데이터 파일 부재 추정).")
#         print("시스템 초기화 완료.")

#     except Exception as e:
#         print(f"시스템 초기화 실패: {e}")
#         raise e

# def structure_template_with_llm(template_string: str) -> StructuredTemplate:
#     parser = JsonOutputParser(pydantic_object=StructuredTemplate)

#     system_prompt = '''당신은 카카오 알림톡 템플릿을 구조화하는 전문가입니다.\n주어진 템플릿 텍스트를 분석하여 'title', 'body', 'buttons' 필드를 가진 JSON 객체로 변환하세요.\n\n# 지시사항:\n1.  'title'은 템플릿의 가장 핵심적인 내용 또는 첫 문장으로, 템플릿의 목적을 명확히 드러내야 합니다.\n2.  'body'는 'title'과 'buttons'를 제외한 템플릿의 모든 본문 내용을 포함해야 합니다. 줄바꿈은 '\\n'으로 유지하세요.\n3.  'buttons'는 템플릿 하단에 있는 버튼들을 파싱하여 [('버튼명', '링크 또는 액션')] 형식의 리스트로 만드세요. 버튼이 없으면 빈 리스트로 두세요.\n4.  템플릿 내에 '#{{변수명}}' 형식의 변수가 있다면 그대로 유지해야 합니다.\n5.  **문맥의 흐름을 파악하여 의미 단위로 문단을 나누고, 줄바꿈(`\\n`)을 추가하세요.**\n6.  나열되는 항목(예: `▶`, `※`)이 있다면 글머리 기호('-')를 사용하여 목록으로 만드세요.\n7.  전체적으로 문장을 간결하고 명확하게 다듬어주세요.\n8.  최종 결과는 반드시 지정된 JSON 형식으로만 출력해야 합니다. 서론이나 추가 설명은 절대 포함하지 마세요.\n'''

#     human_prompt = '''# 실제 작업 요청\n-   **원본 텍스트:** {raw_text}\n-   **출력 형식 (JSON):** {format_instructions}'''

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", human_prompt)
#     ])

#     chain = prompt | llm_fast | parser
#     try:
#         structured_data_dict = chain.invoke({
#             "raw_text": template_string,
#             "format_instructions": parser.get_format_instructions()
#         })
#         return StructuredTemplate(**structured_data_dict)
#     except Exception as e:
#         print(f"Error during structuring template: {e}")
#         return StructuredTemplate(
#             title=template_string.split('\n').strip(),
#             body=template_string,
#             buttons=[]
#         )

# def generate_template_old(request: str, style: str = "기본형") -> ParameterizedResult:
#     def _parameterize_template_internal(template_string: str) -> ParameterizedResult:
#         parser = JsonOutputParser(pydantic_object=ParameterizedResult)
#         prompt = ChatPromptTemplate.from_template(
#             '''당신은 주어진 텍스트를 재사용 가능한 템플릿으로 변환하는 전문가입니다.\n              주어진 텍스트에서 고유명사, 날짜, 장소, 숫자 등 구체적이고 바뀔 수 있는 정보들을 식별하여, 의미 있는 한글 변수명으로 대체해주세요.\n              # 지시사항\n              1. 텍스트의 핵심 정보(누가, 언제, 어디서, 무엇을, 어떻게 등)를 파악합니다.\n              2. 원본 값과 변수명, 그리고 각 변수에 대한 설명을 포함하는 변수 목록을 생성합니다.\n              3. 최종 결과를 지정된 JSON 형식으로만 출력해야 합니다. 그 외의 설명은 절대 추가하지 마세요.\n              # 원본 텍스트:\n              {original_text}\n              # 출력 형식 (JSON):\n              {format_instructions}\n              '''
#         )
#         chain = prompt | llm_fast | parser
#         try:
#             result = chain.invoke({
#                 "original_text": template_string,
#                 "format_instructions": parser.get_format_instructions(),
#             })
#             if not isinstance(result, dict):
#                 result = {"parameterized_template": template_string, "variables": []}
#             if "parameterized_template" not in result:
#                 result["parameterized_template"] = template_string
#             if "variables" not in result:
#                 result["variables"] = []
#             return ParameterizedResult(**result)
#         except Exception as e:
#             print(f"Error during internal parameterization: {e}")
#             return ParameterizedResult(parameterized_template=template_string, variables=[])

#     try:
#         RULES = {
#             "공통": '''\n        - GEN-PREVIEW-001 (미리보기 메시지 제한): 채팅방 리스트와 푸시에 노출되는 문구. 한/영 구분 없이 40자까지 입력 가능. 변수 작성 불가.\n        - GEN-REVIEW-001 (심사 기본 원칙): 알림톡은 정보통신망법과 카카오 내부 기준에 따라 심사되며, 승인된 템플릿만 발송 가능.\n        - GEN-REVIEW-002 (주요 반려 사유): 변수 오류, 과도한 변수(40개 초과) 사용, 변수로만 이루어진 템플릿, 변수가 포함된 버튼명, 변수가 포함된 미리보기 메시지 설정 시 반려됨.\n        - GEN-INFO-DEF-001 (정보성 메시지의 정의): 고객의 요청에 의한 1회성 정보, 거래 확인, 계약 변경 안내 등이 포함됨. 부수적으로 광고가 포함되면 전체가 광고성 정보로 간주됨.\n        - GEN-SERVICE-STD-001 (알림톡 서비스 기준): 알림톡은 수신자에게 반드시 전달되어야 하는 '정형화된 정보성' 메시지에 한함.\n        - GEN-BLACKLIST-001 (블랙리스트 - 포인트/쿠폰): 수신자 동의 없는 포인트 적립/소멸 메시지, 유효기간이 매우 짧은 쿠폰 등은 발송 불가.\n        - GEN-BLACKLIST-002 (블랙리스트 - 사용자 행동 기반): 장바구니 상품 안내, 클릭했던 상품 안내, 생일 축하 메시지, 앱 다운로드 유도 등은 발송 불가.\n        - GEN-GUIDE-001 (정보성/광고성 판단 기준): 특가/할인 상품 안내, 프로모션 또는 이벤트가 혼재된 경우는 광고성 메시지로 판단됨.\n        ''',
#             "기본형": {
#                 "규칙": '''\n        - GEN-TYPE-001 (기본형 특징 및 제한): 고객에게 반드시 전달되어야 하는 정보성 메시지. 한/영 구분 없이 1,000자까지 입력 가능하며, 개인화된 텍스트 영역은 #{변수}로 작성.\n        - GEN-TYPE-002 (부가 정보형 특징 및 제한): 고정적인 부가 정보를 본문 하단에 안내. 최대 500자, 변수 사용 불가, URL 포함 가능. 본문과 합쳐 총 1,000자 초과 불가.\n        - GEN-TYPE-003 (채널추가형 특징 및 제한): 비광고성 메시지 하단에 채널 추가 유도. 안내 멘트는 최대 80자, 변수/URL 포함 불가.\n        ''',
#                 "스타일 가이드": '''\n        # 스타일 설명: 텍스트 중심으로 정보를 전달하는 가장 기본적인 템플릿입니다. 간결하고 직관적인 구성으로 공지, 안내, 상태 변경 등 명확한 내용 전달에 사용됩니다.\n        # 대표 예시 1 (서비스 완료 안내)\n        안녕하세요, #{수신자명}님. 요청하신 #{서비스} 처리가 완료되었습니다. 자세한 내용은 아래 버튼을 통해 확인해주세요.\n        # 대표 예시 2 (예약 리마인드)\n        안녕하세요, #{수신자명}님. 내일(#{예약일시})에 예약하신 서비스가 예정되어 있습니다. 잊지 말고 방문해주세요.\n        '''
#             },
#             "이미지형": {
#                 "규칙": '''\n        - GEN-STYLE-001 (이미지형 특징 및 제한): 포맷화된 정보성 메시지를 시각적으로 안내. 광고성 내용 포함 불가. 템플릿 당 하나의 고정된 이미지만 사용 가능.\n        - GEN-STYLE-002 (이미지형 제작 가이드 - 사이즈): 권장 사이즈는 800x400px (JPG, PNG), 최대 500KB.\n        - GEN-STYLE-009 (이미지 저작권 및 내용 제한): 타인의 지적재산권, 초상권을 침해하는 이미지, 본문과 관련 없는 이미지, 광고성 이미지는 절대 사용 불가.\n        ''',
#                 "스타일 가이드": '''\n        # 스타일 설명: 시각적 요소를 활용하여 사용자의 시선을 끌고 정보를 효과적으로 전달하는 템플릿입니다. 상품 홍보, 이벤트 안내 등 시각적 임팩트가 중요할 때 사용됩니다.\n        # 대표 예시 1 (신상품 출시)\n        (이미지 영역: 새로 출시된 화장품 라인업)\n        '''
#             }
#         }
#         compliance_rules = retrievers.get('compliance').invoke(request)
#         formatted_rules = "\n".join([f"- {doc.metadata.get('rule_id', 'Unknown')}: {doc.page_content}" for doc in compliance_rules])
    
#         prompt = ChatPromptTemplate.from_template(
#              '''당신은 카카오 알림톡 심사 규정을 완벽하게 이해하고 있는 템플릿 제작 전문가입니다.\n### 최종 목표:\n- 사용자의 '최초 요청 의도'를 **최대한 살리면서** 카카오 알림톡의 모든 규정을 통과하는 템플릿 초안을 생성하세요.\n- 만약 요청 내용이 직접적으로 규정을 위반하는 경우, **정보성 메시지로 전환**하여 의도를 유지해야 합니다.\n- **광고성 표현(할인, 쿠폰, 이벤트 등)을 직접 사용하지 않고**, 고객에게 유용한 정보를 제공하는 형태로 표현을 순화하는 것이 핵심입니다.\n\n### 입력 정보:\n- **사용자의 최초 요청:** "{request}"\n- **적용할 스타일:** {style}\n- **스타일 가이드:** {style_guide}\n- **필수 준수 규칙:** {rules}\n\n### 작업 순서:\n1.  **의도 분석:** 사용자의 요청에서 '핵심 의도'가 무엇인지 파악합니다. (예: 추석 맞이 10% 할인을 알리는 것)\n2.  **규정 검토:** 핵심 의도가 '필수 준수 규칙'에 위배되는지 판단합니다.\n3.  **정보성 전환:** 만약 위배된다면, '광고성 표현'을 제거하고, '정보성 메시지'로 전환하여 의도를 간접적으로 전달하는 방법을 모색합니다.\n4.  **변수화:** 변경될 수 있는 정보(고객명, 기간 등)는 '#{{변수명}}' 형식으로 변수화합니다.\n5.  **결과물:** 최종 결과는 수정된 템플릿 텍스트만 출력합니다.\n\n### 템플릿 초안:\n'''
#         )

#         chain = prompt | llm_reasoning| StrOutputParser()
#         generated_template_text = chain.invoke({
#             "request": request,
#             "style": style,
#             "style_guide": RULES.get(style, {}).get("스타일 가이드", ""),
#             "rules": f'{RULES["공통"]}\n{RULES.get(style, {}).get("규칙", "")}\n관련 규칙:\n{formatted_rules}'
#         })
#         return _parameterize_template_internal(generated_template_text.strip())
#     except Exception as e:
#         print(f"Error in generate_template: {e}")
#         return ParameterizedResult(parameterized_template=f"템플릿 생성 중 오류 발생: {request}", variables=[])

# def validate_template_old(template: str) -> Dict:
#     parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
#     relevant_rules = retrievers['compliance'].invoke(template)
#     generation_rules = retrievers['generation'].invoke(template)
#     formatted_rules = "\n".join([f"- {doc.metadata.get('source', 'content')}: {doc.page_content}" for doc in relevant_rules])
#     prompt = ChatPromptTemplate.from_template(
#         '''당신은 카카오 알림톡 심사 가이드라인을 완벽하게 숙지한 AI 심사관입니다.\n          주어진 템플릿이 모든 규칙을 준수하는지 검사하고, 결과를 JSON 형식으로 반환하세요.\n          # 검사할 템플릿:\n          {template}\n          # 주요 심사 규칙:\n          {relevant_rules}\n          {generation_rules}\n          # 지시사항:\n          1. 템플릿이 모든 규칙을 준수하면 status를 "accepted"로 설정합니다.\n          2. 규칙 위반 사항이 하나라도 발견되면 status를 "rejected"로 설정합니다.\n          3. "rejected"인 경우, reason에 어떤 규칙을 위반했는지 명확하고 상세하게 설명합니다.\n          4. evidence 필드에는 위반의 근거가 된 규칙의 content를 정확히 기재합니다.\n          5. 위반 사항을 해결할 수 있는 구체적인 suggestion을 제공합니다.\n          6. 최종 결과는 반드시 지정된 JSON 형식으로만 출력해야 합니다.\n          # 심사 결과 (JSON):\n          {format_instructions}\n          '''
#     )
#     chain = prompt | llm_reasoning | parser
#     try:
#         result = chain.invoke({
#             "template": template,
#             "relevant_rules": formatted_rules,
#             "generation_rules": "\n".join([doc.page_content for doc in generation_rules]),
#             "format_instructions": parser.get_format_instructions()
#         })
#         return result
#     except Exception as e:
#         print(f"Error during validation: {e}")
#         return {"status": "error", "reason": "검증 중 오류 발생"}

# def correct_template(state: dict) -> str:
#     try:
#         attempts = state.get('correction_attempts', 0)

#         # 시도 횟수에 따라 다른 지시사항을 동적으로 설정
#         if attempts == 0:
#             instruction_step = """
#             - 1차 수정: '반려 사유'를 바탕으로 광고성 표현을 정보성 표현으로 순화하세요.
#             """
#         elif attempts == 1:
#             instruction_step = """
#             - 2차 수정: 메시지의 주체를 '사업자'에서 '고객'으로 완전히 전환하여, 고객이 이 메시지를 통해 어떤 유용한 정보를 얻을 수 있는지에 초점을 맞춰 수정하세요.
#             - 수정 원칙: "정보를 전달받는 고객"의 입장에서 유용하고 꼭 필요한 내용만 남기세요.
#             """
#         else: # attempts >= 2
#             instruction_step = """
#             - 최종 수정: '쿠폰', '할인', '이벤트', '특가' 등 직접적인 광고 용어 사용을 금지합니다.
#             - 수정 원칙: 고객에게 필요한 정보 제공 관점으로 표현을 전환하세요. (예: '할인 쿠폰 제공' -> '회원 혜택 안내')
#             """

#         correction_prompt_template = """
#     당신은 카카오 알림톡 심사팀의 수정 전문가입니다. 반려된 템플릿을 수정하는 임무를 수행합니다.\n\n### [임무 목표]\n주어진 '반려 사유'를 완전히 해결하고, '최초 요청 의도({original_request})'를 다시 한번 확인하여 이를 반영한 정보성 템플릿을 완성하세요.\n\n### [분석 정보]\n-   반려된 템플릿 초안:\n    {rejected_draft}\n-   반려 사유: {rejection_reason}\n\n### [수정 지시]\n1.  반려 사유 해결: 반려된 이유를 정확히 파악하여 해당 문제점을 완전히 제거하세요.\n2.  광고성 표현 제거: '할인', '특가', '이벤트', '쿠폰', '혜택'과 같은 광고성 용어를 직접적으로 사용하지 마세요. 대신, 고객에게 유익한 **'정보'**를 제공하는 형태로 표현을 전환하세요.\n3.  관점 전환: 메시지 주체를 '사업자'가 아닌, **'정보를 받는 고객'**의 관점에서 수정하세요.\n4.  가독성 개선: 간결하고 명확한 문체로 다듬고, 필요한 경우 줄바꿈을 추가하여 가독성을 높이세요.\n5.  최종 결과: 수정된 템플릿 내용만 출력합니다.\n\n{instruction_step}\n\n### [수정된 템플릿]\n"""

#         correction_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
        
#         # 동적 지시사항을 프롬프트에 추가
#         correction_prompt = correction_prompt.partial(instruction_step=instruction_step)
        
#         correction_chain = correction_prompt | llm_reasoning | StrOutputParser()

#         rejection_info = state['validation_result']['reason']
#         if state['validation_result'].get('suggestion'):
#             rejection_info += "\n개선 제안: " + state['validation_result']['suggestion']

#         new_draft = correction_chain.invoke({
#             "original_request": state['original_request'],
#             "rejected_draft": state['template_draft'],
#             "rejection_reason": rejection_info
#         })
        
#         # 코드 블록 마크다운을 제거하고 내용만 반환
#         return new_draft.strip().strip('"`')

#     except Exception as e:
#         print(f"Error in correct_template: {e}")
#         traceback.print_exc()
#         return state.get('template_draft', '수정 중 오류가 발생했습니다.')

# def refine_template_with_feedback_old(state: dict) -> str:
#     """
#     사용자의 피드백을 바탕으로 기존 템플릿을 수정합니다.
#     """
#     prompt = ChatPromptTemplate.from_template(
#     """당신은 사용자의 피드백을 반영하여 템플릿을 수정하는 AI 전문가입니다.\n\n          ### 맥락 정보\n          - 사용자의 초기 요청: {initial_request}\n          - 현재 템플릿:\n          {current_template}\n          - 사용자의 수정 요청사항: {user_feedback}\n\n          ### 작업 지시\n          1.  '사용자의 초기 요청' 의도를 잃지 않도록 주의하세요.\n          2.  '현재 템플릿'을 바탕으로 '사용자의 수정 요청사항'을 충실히 반영하여 새로운 템플릿을 만드세요.\n          3.  수정 과정에서 변수 형식('#{{변수명}}')이 깨지지 않도록 유지해야 합니다.\n          4.  최종 결과는 수정된 템플릿 텍스트만 출력해야 합니다. 다른 어떤 설명도 추가하지 마세요.\n\n          ### 수정된 템플릿:\n          """
#     )

#     chain = prompt | llm_reasoning | StrOutputParser()

#     try:
#         refined_template = chain.invoke({
#             "initial_request": state.get('original_request', ''),
#             "current_template": state.get('final_template', ''),
#             "user_feedback": state.get('user_feedback', '')
#         })
#         return refined_template.strip()
#     except Exception as e:
#         print(f"Error during template refinement: {e}")
#         return state.get('final_template', '') # 오류 발생 시 원본 템플릿 반환

# def parameterize_template(template_string: str) -> ParameterizedResult:
#     """
#     주어진 템플릿 문자열에서 변수를 추출하여 파라미터화된 결과를 반환합니다.
#     """
#     parser = JsonOutputParser(pydantic_object=ParameterizedResult)
#     prompt = ChatPromptTemplate.from_template(
#     '''당신은 주어진 텍스트를 재사용 가능한 템플릿으로 변환하는 전문가입니다.\n          주어진 텍스트에서 고유명사, 날짜, 장소, 숫자 등 구체적이고 바뀔 수 있는 정보들을 식별하여, 의미 있는 한글 변수명으로 대체해주세요.\n          # 지시사항\n          1. 텍스트의 핵심 정보(누가, 언제, 어디서, 무엇을 어떻게 등)를 파악합니다.\n          2. 원본 값과 변수명, 그리고 각 변수에 대한 설명을 포함하는 변수 목록을 생성합니다.\n          3. 변수 형식은 #{{변수명}} 이어야 합니다.\n          4. 최종 결과를 지정된 JSON 형식으로만 출력해야 합니다. 그 외의 설명은 절대 추가하지 마세요.\n          # 원본 텍스트:\n          {original_text}\n          # 출력 형식 (JSON):\n          {format_instructions}\n          '''
#     )
#     chain = prompt | llm_fast | parser
#     try:
#         result = chain.invoke({
#             "original_text": template_string,
#             "format_instructions": parser.get_format_instructions(),
#         })
#         if not isinstance(result, dict):
#             result = {"parameterized_template": template_string, "variables": []}
#         if "parameterized_template" not in result:
#             result["parameterized_template"] = template_string
#         if "variables" not in result:
#             result["variables"] = []
#         return ParameterizedResult(**result)
#     except Exception as e:
#         print(f"Error during parameterization: {e}")
#         return ParameterizedResult(parameterized_template=template_string, variables=[])

# async def process_chat_message_async(message: str, state: dict) -> dict:
#     """대화 상태에 따라 비동기로 템플릿 생성 및 수정을 관리 (프론트엔드 호환 버전)"""
#     state = state or {"step": "initial"}
#     step = state.get("step")
#     bot_response: Dict[str, Any] = {}

#     # --- 초기 단계: 유사 템플릿 추천 ---
#     if step == 'initial':
#         if 'original_request' not in state:
#             state['original_request'] = message
#         state['step'] = 'recommend_templates'
        
#         if 'whitelist' not in retrievers or not retrievers['whitelist']:
#             state['step'] = 'select_style_new'
#             bot_response = {
#                 'content': '유사 템플릿 검색 기능이 비활성화 상태입니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:', 
#                 'options': ["기본형", "이미지형", "강조형", "아이템 리스트형", "복합형"]
#             }
#             return {"response": bot_response, "state": state}
        
#         similar_docs = retrievers['whitelist'].invoke(state['original_request'])
#         if not similar_docs:
#             state['step'] = 'select_style_new'
#             bot_response = {
#                 'content': '유사한 기존 템플릿을 찾지 못했습니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:', 
#                 'options': ["기본형", "이미지형", "강조형", "아이템 리스트형", "복합형"]
#             }
#             return {"response": bot_response, "state": state}
        
#         structured_templates = [st.model_dump() for st in [structure_template_with_llm(doc.page_content) for doc in similar_docs[:3]]]
#         state['retrieved_similar_templates'] = [doc.page_content for doc in similar_docs[:3]]
        
#         bot_response = {
#             'content': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다. 이 템플릿을 사용하시거나 새로 만드시겠어요?',
#             'templates': structured_templates
#         }

#     # --- 추천된 템플릿 사용 또는 새로 만들기 선택 ---
#     elif step == 'recommend_templates':
#         if message.startswith('템플릿 ') and message.endswith(' 사용'):
#             try:
#                 template_idx = int(message.split()[1]) - 1
#                 selected_template = state['retrieved_similar_templates'][template_idx]
                
#                 parameterized_result = parameterize_template(selected_template)
                
#                 state["base_template"] = parameterized_result.parameterized_template
#                 state["template_draft"] = parameterized_result.parameterized_template
#                 state["variables_info"] = [v.model_dump() for v in parameterized_result.variables]
                
#                 state['step'] = 'completed_new_style'
#                 return await process_chat_message_async(message, state)
#             except (IndexError, ValueError):
#                 pass
        
#         elif message == '새로 만들기':
#             state['step'] = 'select_style_new'
#             bot_response = {
#                 'content': '새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요.',
#                 'options': ["기본형", "이미지형", "강조형", "아이템 리스트형", "복합형"]
#             }
#         else:
#             options = [f'템플릿 {i+1} 사용' for i in range(len(state.get('retrieved_similar_templates',[])))] + ['새로 만들기']
#             bot_response = {'content': '제시된 옵션 중에서 선택해주세요.', 'options': options}

#     # --- [개선된 파이프라인] 스타일 선택 및 즉시 생성 ---
#     elif step == "select_style_new":
#         STYLE_OPTIONS = ["기본형", "이미지형", "강조형", "아이템 리스트형", "복합형"]
#         if message in STYLE_OPTIONS:
#             state["selected_style"] = message
#             try:
#                 original_request = state.get('original_request', '')
#                 if not original_request:
#                     state['step'] = 'initial'
#                     bot_response = {'content': '오류: 최초 요청을 찾을 수 없습니다. 처음부터 다시 시작해주세요.'}
#                     return {"response": bot_response, "state": state}

#                 template_obj = await generate_template_for_style(original_request, state["selected_style"])
#                 validation_result = await validate_template_new(template_obj.model_dump_json())

#                 if validation_result.status == "rejected":
#                     raise ValueError(f"생성된 템플릿이 내부 규정에 맞지 않습니다: {validation_result.reason}")

#                 response_model = TemplateResponse(style=state["selected_style"], template_data=template_obj)
#                 state["final_template_response"] = response_model.model_dump()
#                 state["step"] = "awaiting_feedback_new"

#                 frontend_data = _convert_template_response_to_frontend_format(response_model)
                
#                 bot_response = {
#                     "content": "✅ 템플릿 생성이 완료되었습니다. 수정하고 싶은 부분이 있다면 말씀해주세요. 없다면 '완료'라고 입력해주세요.",
#                     "options": ["완료"],
#                     **frontend_data
#                 }
#             except Exception as e:
#                 print(f"Error during generation: {e}")
#                 state["step"] = "select_style_new"
#                 bot_response = {
#                     "content": f"⚠️ 생성 중 오류가 발생했습니다: {e}\n내용을 조금 더 구체적으로 작성하여 다시 요청해주세요.",
#                     "options": STYLE_OPTIONS
#                 }
#         else:
#             bot_response = {"content": "잘못된 스타일입니다. 제시된 옵션 중에서 선택해주세요.", "options": STYLE_OPTIONS}

#     # --- 생성된 템플릿에 대한 피드백 대기 및 처리 ---
#     elif step == "awaiting_feedback_new":
#         if message == "완료":
#             # 프론트엔드가 '저장 버튼'을 활성화할 수 있도록 메시지 변경
#             final_response_model = TemplateResponse(**state["final_template_response"])
#             frontend_data = _convert_template_response_to_frontend_format(final_response_model)
#             bot_response = {
#                 "content": "템플릿 생성을 마칩니다. 우측의 '이 템플릿 저장하기' 버튼을 눌러 저장해주세요.",
#                 **frontend_data
#             }
#             # isConversationComplete 플래그 추가
#             bot_response['isConversationComplete'] = True 
#             state.clear()
#             state["step"] = "initial"
#         else:
#             try:
#                 refined_data_dict = await refine_template_with_feedback_new(state, message)
                
#                 current_response = TemplateResponse(**state["final_template_response"])
#                 original_template_obj = current_response.template_data
#                 refined_obj = type(original_template_obj)(**refined_data_dict)
                
#                 current_response.template_data = refined_obj
#                 state["final_template_response"] = current_response.model_dump()

#                 frontend_data = _convert_template_response_to_frontend_format(current_response)

#                 bot_response = {
#                     "content": "수정이 완료되었습니다. 더 수정할 부분이 있나요?",
#                     "options": ["완료"],
#                     **frontend_data
#                 }
#             except Exception as e:
#                 print(f"Error during refinement: {e}")
#                 bot_response = {"content": f"⚠️ 수정 중 오류가 발생했습니다: {e}\n다른 방식으로 다시 요청해주세요."}

#     # --- [기존 코드 흐름] 유사 템플릿 선택 후 완료 단계 ---
#     elif step == 'completed_new_style':
#         final_template_text = state['base_template']
#         variables_info = state.get('variables_info', [])
        
#         variables = [{"name": v['name'], "type": "string", "example": v['original_value']} for v in variables_info]
#         editable_variables = {"parameterized_template": final_template_text, "variables": variables}
        
#         parts = final_template_text.split('\n\n', 1)
#         title = parts[0]
#         body = parts[1] if len(parts) > 1 else ""
#         structured_template = {"title": title, "body": body}

#         var_text = "\n".join([f"- {var['name']} (원본: {var['original_value']}): {var['description']}" for var in variables_info])
        
#         bot_response = {
#             "content": f"✅ 최종 템플릿이 완성되었습니다!\n\n```\n{final_template_text}\n```\n\n추출된 변수 정보:\n{var_text}\n\n이 템플릿을 사용하시겠습니까? 아니면 추가 수정이 필요하신가요?",
#             "template": final_template_text,
#             "structured_template": structured_template,
#             "templates": [structured_template],
#             "editable_variables": editable_variables,
#             "options": ['사용', '수정']
#         }
#         state['step'] = 'final_review_new_style'
    
#     # --- [기존 코드 흐름] 최종 검토 (사용/수정) ---
#     elif state['step'] == 'final_review_new_style':
#         if message == '사용':
#             final_template_text = state['base_template']
#             bot_response = {'content': '템플릿이 최종 확정되었습니다. 새로운 템플릿을 만들려면 아무 내용이나 입력해주세요.'}
#             bot_response['isConversationComplete'] = True
#             state.clear()
#             state['step'] = 'initial'
#         elif message == '수정':
#             state['step'] = 'awaiting_user_feedback_old_style'
#             bot_response = {'content': f'어떤 부분을 수정하시겠어요? 구체적으로 알려주세요.\n\n현재 템플릿:\n```{state["base_template"]}```'}
#         else:
#             bot_response = {'content': "잘못된 입력입니다. '사용' 또는 '수정'으로만 답해주세요.", 'options': ['사용', '수정']}

#     # --- [기존 코드 흐름] 사용자 피드백 대기 및 처리 ---
#     elif state['step'] == 'awaiting_user_feedback_old_style':
#         state['user_feedback'] = message
#         state['final_template'] = state['base_template']
#         refined_template = refine_template_with_feedback_old(state)
        
#         state['base_template'] = refined_template
#         state['template_draft'] = refined_template
        
#         validation_result = validate_template_old(refined_template)
#         state['validation_result'] = validation_result

#         if validation_result['status'] == 'accepted':
#             state['step'] = 'completed_new_style'
#             return await process_chat_message_async("", state)
#         else:
#             state['step'] = 'correction_after_feedback_old_style'
#             bot_response = {'content': f'수정된 템플릿에 문제가 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result.get("suggestion", "없음")}\n\nAI가 다시 수정하거나 직접 수정하시겠습니까?', 'options': ['AI 수정', '직접 수정']}
    
#     # --- 이하 기존 코드 흐름에 대한 처리 (생략 없이 모두 포함) ---
#     elif state['step'] == 'correction_after_feedback_old_style':
#         if message == 'AI 수정':
#             state['correction_attempts'] = 0
#             state['step'] = 'correction_old_style'
#             return await process_chat_message_async("", state)
#         elif message == '직접 수정':
#             state['step'] = 'awaiting_user_correction_old_style'
#             bot_response = {'content': f'현재 템플릿 초안입니다. 직접 수정할 내용을 입력해주세요.\n\n```{state["template_draft"]}```'}
#         else:
#             bot_response = {'content': "잘못된 입력입니다. 'AI 수정' 또는 '직접 수정'으로만 답해주세요.", 'options': ['AI 수정', '직접 수정']}

#     elif state['step'] == 'correction_old_style':
#         if state.get('correction_attempts', 0) < MAX_CORRECTION_ATTEMPTS:
#             state['correction_attempts'] = state.get('correction_attempts', 0) + 1
#             corrected_template = correct_template(state)
#             validation_result = validate_template_old(corrected_template)
#             state["validation_result"] = validation_result
            
#             if validation_result["status"] == "accepted":
#                 state['base_template'] = corrected_template
#                 state["template_draft"] = corrected_template
#                 state["step"] = "completed_new_style"
#                 return await process_chat_message_async(message, state)
#             else:
#                 state['template_draft'] = corrected_template
#                 bot_response = {'content': f'AI 자동 수정 후에도 문제가 발견되었습니다. (시도: {state["correction_attempts"]}/{MAX_CORRECTION_ATTEMPTS})\n\n문제점: {validation_result["reason"]}\n\nAI가 다시 한번 수정합니다.'}
#         else:
#             state['step'] = 'manual_correction_old_style'
#             bot_response = {'content': f'AI 자동 수정({MAX_CORRECTION_ATTEMPTS}회)에 실패했습니다. 직접 수정하시겠습니까?', 'options': ['예', '아니오']}

#     elif state['step'] == 'manual_correction_old_style':
#         if message == '예':
#             state['step'] = 'awaiting_user_correction_old_style'
#             bot_response = {'content': f'현재 템플릿 초안입니다. 직접 수정할 내용을 입력해주세요.\n\n```{state["template_draft"]}```'}
#         elif message == '아니오':
#             state['step'] = 'initial'
#             bot_response = {'content': '템플릿 생성을 취소하고 초기 상태로 돌아갑니다.'}
#         else:
#             bot_response = {'content': "잘못된 입력입니다. '예' 또는 '아니오'로만 답해주세요.", 'options': ['예', '아니오']}

#     elif state['step'] == 'awaiting_user_correction_old_style':
#         state['template_draft'] = message
#         validation_result = validate_template_old(state['template_draft'])
#         state['validation_result'] = validation_result

#         if validation_result['status'] == 'accepted':
#             state['base_template'] = state['template_draft']
#             state['step'] = 'completed_new_style'
#             return await process_chat_message_async(message, state)
#         else:
#             state['step'] = 'manual_correction_feedback_old_style'
#             bot_response = {'content': f'수정하신 템플릿에 문제가 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result.get("suggestion", "없음")}\n\n다시 수정하시겠습니까?', 'options': ['예', '아니오']}

#     elif state['step'] == 'manual_correction_feedback_old_style':
#         if message == '예':
#             state['step'] = 'awaiting_user_correction_old_style'
#             bot_response = {'content': f'현재 템플릿 초안입니다. 다시 수정할 내용을 입력해주세요.\n\n```{state["template_draft"]}```'}
#         elif message == '아니오':
#             state['step'] = 'initial'
#             bot_response = {'content': '템플릿 생성을 취소하고 초기 상태로 돌아갑니다.'}
#         else:
#             bot_response = {'content': "잘못된 입력입니다. '예' 또는 '아니오'로만 답해주세요.", 'options': ['예', '아니오']}

#     # --- 예외 처리 및 최종 반환 ---
#     if not bot_response:
#         bot_response = {'content': '알 수 없는 오류가 발생했습니다. 대화를 다시 시작해주세요.'}
#         state = {'step': 'initial'}

#     # 최종적으로 프론트엔드가 받을 형식으로 패키징
#     return {
#         "response": bot_response,
#         "state": state
#     }


# def _convert_template_response_to_frontend_format(response_obj: TemplateResponse) -> dict:
#     """
#     Pydantic TemplateResponse 객체를 프론트엔드의 BotResponse 형식에 맞게 변환합니다.
#     'structured_template', 'template', 'editable_variables'를 생성합니다.
#     """
#     if not response_obj or not response_obj.template_data:
#         return {}

#     # 1. structured_template 생성 (Pydantic 모델을 dict로 변환)
#     structured_template = response_obj.template_data.model_dump()

#     # 2. template (raw text) 생성
#     # body 필드만 사용하거나, 필요에 따라 title 등을 조합할 수 있습니다.
#     # 여기서는 body를 중심으로 생성합니다.
#     body = structured_template.get('body', '')
#     raw_template = body.strip()

#     # 3. editable_variables 추출
#     # 정규식을 사용하여 #{{변수명}} 형식의 모든 변수를 찾습니다.
#     variable_names = re.findall(r'#\{\{([^}]+)\}\}', raw_template)
    
#     # 중복을 제거하고 프론트엔드 형식에 맞게 변환합니다.
#     variables = [
#         {"name": name, "type": "string", "example": f"{name} 예시"}
#         for name in sorted(list(set(variable_names)))
#     ]

#     editable_variables = {
#         "parameterized_template": raw_template,
#         "variables": variables
#     }

#     return {
#         "structured_template": structured_template,
#         "templates": [structured_template], # 프론트엔드는 배열을 기대하므로 배열로 감싸줍니다.
#         "template": raw_template,
#         "editable_variables": editable_variables,
#     }

# def process_chat_message(message: str, state: dict) -> dict:
#     """비동기 함수를 동기적으로 실행하기 위한 래퍼"""
#     try:
#         # 이미 실행 중인 이벤트 루프가 있는지 확인 (Jupyter, FastAPI 등 환경)
#         loop = asyncio.get_event_loop()
#         if loop.is_running():
#             # Nesting을 피하기 위해 새 루프에서 실행하는 것이 더 안정적일 수 있음
#             # 여기서는 간단하게 처리
#             future = asyncio.run_coroutine_threadsafe(process_chat_message_async(message, state), loop)
#             return future.result()
#         else:
#             return asyncio.run(process_chat_message_async(message, state))
#     except RuntimeError:
#         # 실행 중인 루프가 없는 경우 (일반적인 스크립트 실행)
#         return asyncio.run(process_chat_message_async(message, state))


import os
import json
import re
import asyncio
from typing import TypedDict, List, Optional, Dict, Any, Literal, Union
import sys
import traceback

# Pydantic 및 LangChain 호환성을 위한 임포트
from pydantic import BaseModel, Field, PrivateAttr

# LangChain 및 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers.pydantic import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.callbacks import Callbacks

# FlashRank 임포트
try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    print("FlashRank 또는 관련 모듈을 찾을 수 없습니다. Reranking 기능이 비활성화됩니다.")
    Ranker = None

# .env 파일에서 API 키 로드
load_dotenv()

# --- 설정 및 모델 정의 ---
MAX_CORRECTION_ATTEMPTS = 3

# --- 전역 변수 및 헬퍼 함수 ---
llm_reasoning = None # gpt-5 (고성능 추론용)
llm_fast = None      # gpt-4.1 (단순 작업용)
llm_medium = None    # gpt-4.1-mini (단순 작업용)
llm_general = None   # gpt-4o (새 코드에서 사용)
retrievers = {}
approved_templates = []
rejected_templates = []

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
    buttons: Optional[List[tuple[str, str]]] = Field(None, description="템플릿에 포함될 버튼 리스트. 예: [('웹사이트', '자세히 보기')]"
)

class Button(BaseModel):
    """템플릿 하단 버튼 모델"""
    type: Literal["AL", "WL", "AC", "BK", "MD"] = Field(description="AL: 앱링크, WL: 웹링크, AC: 채널추가, BK: 봇키워드, MD: 메시지전달")
    name: str = Field(max_length=28, description="버튼 이름 (최대 28자)")
    value: str = Field(description="URL, 앱 경로, 봇키워드 등")

class Highlight(BaseModel):
    """강조표기형 타이틀/서브타이틀 모델"""
    title: str = Field(max_length=23, description="강조형 타이틀 (최대 23자)")
    subtitle: str = Field(max_length=18, description="강조형 서브타이틀 (변수 사용 불가, 최대 18자)")

class Item(BaseModel):
    """아이템 리스트 개별 항목 모델"""
    name: str = Field(max_length=6, description="아이템명 (변수 사용 불가, 최대 6자)")
    description: str = Field(max_length=23, description="설명 (변수 사용 가능, 최대 23자)")
    summary: Optional[str] = Field(None, description="우측 요약 정보 (숫자, 통화기호 등만 가능)")

class ItemHighlight(BaseModel):
    """아이템 리스트 하이라이트 모델"""
    thumbnail_required: bool = Field(default=True, description="썸일 필요 여부")
    text: str = Field(max_length=21, description="하이라이트 텍스트 (최대 21자)")
    description: str = Field(max_length=13, description="하이라이트 설명 (최대 13자)")

class BasicTemplate(BaseModel):
    """기본형 템플릿 모델"""
    body: str = Field(max_length=1300, description="본문 (최대 1300자)")
    footer: Optional[str] = Field(None, max_length=500, description="부가 정보 (최대 500자, 변수 불가)")
    add_channel: Optional[bool] = Field(False, description="채널 추가 버튼 여부")
    buttons: List[Button] = Field([], max_items=5, description="버튼 리스트 (최대 5개)")

class ImageTemplate(BasicTemplate):
    """이미지형 템플릿 모델 (기본형 상속)"""
    image_url: str = Field(description="이미지 URL (800x400px 권장)")

class HighlightTemplate(BasicTemplate):
    """강조표기형 템플릿 모델 (기본형 상속)"""
    highlight: Highlight = Field(description="강조형 타이틀/서브타이틀")

class ItemListTemplate(BaseModel):
    """아이템 리스트형 템플릿 모델"""
    header: Optional[str] = Field(None, max_length=16, description="헤더 (최대 16자)")
    item_highlight: Optional[ItemHighlight] = Field(None)
    body: str = Field(max_length=1300)
    items: List[Item] = Field(min_items=2, max_items=10)
    buttons: List[Button] = Field([], max_items=5)

class CompositeTemplate(BasicTemplate):
    """복합형 템플릿 모델 (기본형 상속)"""
    footer: str = Field(..., max_length=500, description="부가 정보 (필수)")
    add_channel: bool = Field(True, description="채널 추가 (필수)")

class TemplateResponse(BaseModel):
    """최종 생성 결과 응답 모델"""
    style: Literal["기본형", "이미지형", "강조형", "아이템 리스트형", "복합형"]
    template_data: Union[BasicTemplate, ImageTemplate, HighlightTemplate, ItemListTemplate, CompositeTemplate]

class ValidationResult(BaseModel):
    """내부 검증 결과 모델"""
    status: Literal["accepted", "rejected"]
    reason: Optional[str] = None


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


async def _create_and_run_chain(request: str, pydantic_model: BaseModel, system_prompt: str):
    """LLM 체인을 생성하고 비동기로 실행하는 헬퍼 함수"""
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("사용자 요청: {request}"),
        ],
        input_variables=["request"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm_general | parser
    return await chain.ainvoke({"request": request})

async def generate_template_for_style(request: str, style: str) -> BaseModel:
    """스타일에 맞는 생성기를 호출하여 Pydantic 객체를 생성"""
    style_configs = {
        "기본형": (BasicTemplate, "텍스트 중심의 표준 템플릿. 본문, 푸터, 버튼으로 구성."),
        "이미지형": (ImageTemplate, "상단에 이미지가 포함된 템플릿. `image_url`은 적절한 예시 URL로 채워야 함."),
        "강조형": (HighlightTemplate, "핵심 정보를 `highlight` 객체로 추출하여 주목도를 높이는 템플릿."),
        "아이템 리스트형": (ItemListTemplate, "주문/예약 내역 등 반복 정보를 목록 형태로 보여주는 템플릿. `header`, `item_highlight`, `items` 등 모든 요소를 충실히 생성."),
        "복합형": (CompositeTemplate, "`footer`와 `add_channel`이 필수로 포함되는 템플릿.")
    }
    
    if style not in style_configs:
        raise ValueError(f"지원하지 않는 스타일입니다: {style}")

    pydantic_model, style_desc = style_configs[style]
    
    system_prompt = f"""당신은 카카오 알림톡 '{style}' 템플릿 제작 전문가입니다.
사용자 요청을 분석하여 '{style}' 스타일에 맞는 완벽한 JSON 객체를 생성하세요.
{style_desc}
변수가 필요한 곳은 '#{{{{변수명}}}}' 형식을 사용하고, 모든 필드는 카카오 가이드라인 제약 조건(글자 수 등)을 엄격히 준수해야 합니다.

# 출력 형식
{{format_instructions}}"""
    return await _create_and_run_chain(request, pydantic_model, system_prompt)


async def validate_template_new(template_json: str) -> ValidationResult:
    """생성된 템플릿을 검증하는 로직 (새 코드의 validate_template)"""
    # 실제 구현 시: RAG로 규정 조회 후 LLM으로 위반 여부 판단
    print("🤖 템플릿 내부 검증 수행...")
    # 여기서는 항상 accepted를 반환하지만, 실제 구현에서는 LLM을 통해 검증 로직을 추가해야 합니다.
    return ValidationResult(status="accepted")

async def refine_template_with_feedback_new(state: dict, feedback: str) -> dict:
    """피드백과 전체 대화 맥락을 반영하여 템플릿(Pydantic 객체)을 수정합니다."""
    
    # 1. state에서 맥락 정보 추출
    initial_request = state.get("original_request", "알 수 없음")
    current_response = TemplateResponse(**state["final_template_response"])
    original_template_obj = current_response.template_data
    
    # Pydantic 객체를 딕셔너리와 JSON 문자열로 변환
    template_dict = original_template_obj.model_dump()
    template_json_str = json.dumps(template_dict, ensure_ascii=False, indent=2)
    
    # AI가 참고할 수 있도록 현재 템플릿의 본문을 텍스트로 추출
    current_template_text = template_dict.get("body", "본문 없음")

    # 2. Pydantic 파서와 개선된 프롬프트 정의
    parser = PydanticOutputParser(pydantic_object=type(original_template_obj))
    
    system_prompt = """당신은 사용자의 피드백을 반영하여 JSON 형식의 템플릿을 수정하는 AI 전문가입니다.

### 맥락 정보
- **사용자의 최초 요청 의도:** {initial_request}
- **수정 전 템플릿 원본 텍스트:** ```
  {current_template_text}
사용자의 수정 요청사항: {feedback}

작업 지시
'맥락 정보'를 종합적으로 이해하여 사용자의 수정 의도를 명확히 파악하세요.

사용자의 요청이 "이 부분", "저기" 등 모호하더라도, '수정 전 템플릿 원본 텍스트'를 참고하여 수정 대상을 정확히 추론해야 합니다.

사용자가 '제거' 또는 '삭제'를 요청하면, 다른 지시보다 우선하여 해당 내용을 반드시 제거해야 합니다.

'최초 요청 의도'를 잃지 않으면서, 위 지시사항을 바탕으로 아래 '수정 대상 JSON'을 수정하세요.

최종 결과는 반드시 수정된 JSON 객체만 출력해야 합니다. 서론이나 부가 설명은 절대 포함하지 마세요.

출력 형식
{format_instructions}
"""

    human_prompt = """### 수정 대상 JSON
JSON

{template_json}
사용자의 수정 요청사항
{feedback}

수정된 템플릿 (JSON):
"""

    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# 3. 체인 생성 및 실행
    chain = prompt | llm_general | parser

    refined_obj = await chain.ainvoke({ 
    "initial_request": initial_request,
    "current_template_text": current_template_text,
    "template_json": template_json_str, 
    "feedback": feedback,
    "format_instructions": parser.get_format_instructions()
})

    return refined_obj.model_dump()

def initialize_system():
    global llm_reasoning, llm_fast, llm_medium, llm_general, retrievers, approved_templates, rejected_templates
    if llm_reasoning is not None:
        return

    print("서버 시작: 시스템 초기화를 진행합니다...")
    try:
        data_dir = 'data'
    
        llm_reasoning = ChatOpenAI(model="gpt-4.1", temperature=0.3)
        llm_medium = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
        llm_fast = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
        llm_general = ChatOpenAI(model="gpt-4o", temperature=0.1) # 새 코드에서 사용

    
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
        approved_templates = load_line_by_line(os.path.join(data_dir, "approved_templates.txt"))
        rejected_templates = load_by_separator(os.path.join(data_dir, "rejected_templates.txt"))
    
        docs_compliance = CustomRuleLoader(os.path.join(data_dir, "compliance_rules.txt")).load()
        docs_generation = CustomRuleLoader(os.path.join(data_dir, "generation_rules.txt")).load()
        docs_whitelist = [Document(page_content=t) for t in approved_templates]
        docs_rejected = [Document(page_content=t) for t in rejected_templates]
    
        def create_db(name, docs):
            if not docs:
                print(f"🚨 '{name}'에 대한 문서가 없어 DB 생성을 건너뜁니다.")
                return None
        
            print(f"✨ '{name}' 컬렉션을 인메모리 DB에 생성합니다...")
            db = Chroma.from_documents(
                docs, 
                embeddings, 
                collection_name=name
            )
            print(f"💾 '{name}' 컬렉션이 인메모리 DB에 저장되었습니다.")
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
                print(f"🚨 경고: '{name}' 리트리버를 생성하지 못했습니다 (관련 데이터 파일 부재 추정)."
)
        print("시스템 초기화 완료.")

    except Exception as e:
        print(f"시스템 초기화 실패: {e}")
        raise e

def structure_template_with_llm(template_string: str) -> StructuredTemplate:
    parser = JsonOutputParser(pydantic_object=StructuredTemplate)

    system_prompt = '''당신은 카카오 알림톡 템플릿을 구조화하는 전문가입니다.\n주어진 템플릿 텍스트를 분석하여 'title', 'body', 'buttons' 필드를 가진 JSON 객체로 변환하세요.\n\n# 지시사항:\n1.  'title'은 템플릿의 가장 핵심적인 내용 또는 첫 문장으로, 템플릿의 목적을 명확히 드러내야 합니다.\n2.  'body'는 'title'과 'buttons'를 제외한 템플릿의 모든 본문 내용을 포함해야 합니다. 줄바꿈은 '\\n'으로 유지하세요.\n3.  'buttons'는 템플릿 하단에 있는 버튼들을 파싱하여 [('버튼명', '링크 또는 액션')] 형식의 리스트로 만드세요. 버튼이 없으면 빈 리스트로 두세요.\n4.  템플릿 내에 '#{{변수명}}' 형식의 변수가 있다면 그대로 유지해야 합니다.\n5.  **문맥의 흐름을 파악하여 의미 단위로 문단을 나누고, 줄바꿈(`\\n`)을 추가하세요.**\n6.  나열되는 항목(예: `▶`, `※`)이 있다면 글머리 기호('-')를 사용하여 목록으로 만드세요.\n7.  전체적으로 문장을 간결하고 명확하게 다듬어주세요.\n8.  최종 결과는 반드시 지정된 JSON 형식으로만 출력해야 합니다. 서론이나 추가 설명은 절대 포함하지 마세요.\n'''

    human_prompt = '''# 실제 작업 요청\n-   **원본 텍스트:** {raw_text}\n-   **출력 형식 (JSON):** {format_instructions}'''

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    chain = prompt | llm_fast | parser
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

def generate_template_old(request: str, style: str = "기본형") -> ParameterizedResult:
    def _parameterize_template_internal(template_string: str) -> ParameterizedResult:
        parser = JsonOutputParser(pydantic_object=ParameterizedResult)
        prompt = ChatPromptTemplate.from_template(
            '''당신은 주어진 텍스트를 재사용 가능한 템플릿으로 변환하는 전문가입니다.\n              주어진 텍스트에서 고유명사, 날짜, 장소, 숫자 등 구체적이고 바뀔 수 있는 정보들을 식별하여, 의미 있는 한글 변수명으로 대체해주세요.\n              # 지시사항\n              1. 텍스트의 핵심 정보(누가, 언제, 어디서, 무엇을, 어떻게 등)를 파악합니다.\n              2. 원본 값과 변수명, 그리고 각 변수에 대한 설명을 포함하는 변수 목록을 생성합니다.\n              3. 최종 결과를 지정된 JSON 형식으로만 출력해야 합니다. 그 외의 설명은 절대 추가하지 마세요.\n              # 원본 텍스트:\n              {original_text}\n              # 출력 형식 (JSON):\n              {format_instructions}\n              '''
        )
        chain = prompt | llm_fast | parser
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
            return ParameterizedResult(**result)
        except Exception as e:
            print(f"Error during internal parameterization: {e}")
            return ParameterizedResult(parameterized_template=template_string, variables=[])

    try:
        RULES = {
            "공통": '''\n        - GEN-PREVIEW-001 (미리보기 메시지 제한): 채팅방 리스트와 푸시에 노출되는 문구. 한/영 구분 없이 40자까지 입력 가능. 변수 작성 불가.\n        - GEN-REVIEW-001 (심사 기본 원칙): 알림톡은 정보통신망법과 카카오 내부 기준에 따라 심사되며, 승인된 템플릿만 발송 가능.\n        - GEN-REVIEW-002 (주요 반려 사유): 변수 오류, 과도한 변수(40개 초과) 사용, 변수로만 이루어진 템플릿, 변수가 포함된 버튼명, 변수가 포함된 미리보기 메시지 설정 시 반려됨.\n        - GEN-INFO-DEF-001 (정보성 메시지의 정의): 고객의 요청에 의한 1회성 정보, 거래 확인, 계약 변경 안내 등이 포함됨. 부수적으로 광고가 포함되면 전체가 광고성 정보로 간주됨.\n        - GEN-SERVICE-STD-001 (알림톡 서비스 기준): 알림톡은 수신자에게 반드시 전달되어야 하는 '정형화된 정보성' 메시지에 한함.\n        - GEN-BLACKLIST-001 (블랙리스트 - 포인트/쿠폰): 수신자 동의 없는 포인트 적립/소멸 메시지, 유효기간이 매우 짧은 쿠폰 등은 발송 불가.\n        - GEN-BLACKLIST-002 (블랙리스트 - 사용자 행동 기반): 장바구니 상품 안내, 클릭했던 상품 안내, 생일 축하 메시지, 앱 다운로드 유도 등은 발송 불가.\n        - GEN-GUIDE-001 (정보성/광고성 판단 기준): 특가/할인 상품 안내, 프로모션 또는 이벤트가 혼재된 경우는 광고성 메시지로 판단됨.\n        ''',
            "기본형": {
                "규칙": '''\n        - GEN-TYPE-001 (기본형 특징 및 제한): 고객에게 반드시 전달되어야 하는 정보성 메시지. 한/영 구분 없이 1,000자까지 입력 가능하며, 개인화된 텍스트 영역은 #{변수}로 작성.\n        - GEN-TYPE-002 (부가 정보형 특징 및 제한): 고정적인 부가 정보를 본문 하단에 안내. 최대 500자, 변수 사용 불가, URL 포함 가능. 본문과 합쳐 총 1,000자 초과 불가.\n        - GEN-TYPE-003 (채널추가형 특징 및 제한): 비광고성 메시지 하단에 채널 추가 유도. 안내 멘트는 최대 80자, 변수/URL 포함 불가.\n        ''',
                "스타일 가이드": '''\n        # 스타일 설명: 텍스트 중심으로 정보를 전달하는 가장 기본적인 템플릿입니다. 간결하고 직관적인 구성으로 공지, 안내, 상태 변경 등 명확한 내용 전달에 사용됩니다.\n        # 대표 예시 1 (서비스 완료 안내)\n        안녕하세요, #{수신자명}님. 요청하신 #{서비스} 처리가 완료되었습니다. 자세한 내용은 아래 버튼을 통해 확인해주세요.\n        # 대표 예시 2 (예약 리마인드)\n        안녕하세요, #{수신자명}님. 내일(#{예약일시})에 예약하신 서비스가 예정되어 있습니다. 잊지 말고 방문해주세요.\n        '''
            },
            "이미지형": {
                "규칙": '''\n        - GEN-STYLE-001 (이미지형 특징 및 제한): 포맷화된 정보성 메시지를 시각적으로 안내. 광고성 내용 포함 불가. 템플릿 당 하나의 고정된 이미지만 사용 가능.\n        - GEN-STYLE-002 (이미지형 제작 가이드 - 사이즈): 권장 사이즈는 800x400px (JPG, PNG), 최대 500KB.\n        - GEN-STYLE-009 (이미지 저작권 및 내용 제한): 타인의 지적재산권, 초상권을 침해하는 이미지, 본문과 관련 없는 이미지, 광고성 이미지는 절대 사용 불가.\n        ''',
                "스타일 가이드": '''\n        # 스타일 설명: 시각적 요소를 활용하여 사용자의 시선을 끌고 정보를 효과적으로 전달하는 템플릿입니다. 상품 홍보, 이벤트 안내 등 시각적 임팩트가 중요할 때 사용됩니다.\n        # 대표 예시 1 (신상품 출시)\n        (이미지 영역: 새로 출시된 화장품 라인업)\n        '''
            }
        }
        compliance_rules = retrievers.get('compliance').invoke(request)
        formatted_rules = "\n".join([f"- {doc.metadata.get('rule_id', 'Unknown')}: {doc.page_content}" for doc in compliance_rules])
    
        prompt = ChatPromptTemplate.from_template(
             '''당신은 카카오 알림톡 심사 규정을 완벽하게 이해하고 있는 템플릿 제작 전문가입니다.\n### 최종 목표:\n- 사용자의 '최초 요청 의도'를 **최대한 살리면서** 카카오 알림톡의 모든 규정을 통과하는 템플릿 초안을 생성하세요.\n- 만약 요청 내용이 직접적으로 규정을 위반하는 경우, **정보성 메시지로 전환**하여 의도를 유지해야 합니다.\n- **광고성 표현(할인, 쿠폰, 이벤트 등)을 직접 사용하지 않고**, 고객에게 유용한 정보를 제공하는 형태로 표현을 순화하는 것이 핵심입니다.\n\n### 입력 정보:\n- **사용자의 최초 요청:** "{request}"\n- **적용할 스타일:** {style}\n- **스타일 가이드:** {style_guide}\n- **필수 준수 규칙:** {rules}\n\n### 작업 순서:\n1.  **의도 분석:** 사용자의 요청에서 '핵심 의도'가 무엇인지 파악합니다. (예: 추석 맞이 10% 할인을 알리는 것)\n2.  **규정 검토:** 핵심 의도가 '필수 준수 규칙'에 위배되는지 판단합니다.\n3.  **정보성 전환:** 만약 위배된다면, '광고성 표현'을 제거하고, '정보성 메시지'로 전환하여 의도를 간접적으로 전달하는 방법을 모색합니다.\n4.  **변수화:** 변경될 수 있는 정보(고객명, 기간 등)는 '#{{변수명}}' 형식으로 변수화합니다.\n5.  **결과물:** 최종 결과는 수정된 템플릿 텍스트만 출력합니다.\n\n### 템플릿 초안:\n'''
        )

        chain = prompt | llm_reasoning| StrOutputParser()
        generated_template_text = chain.invoke({
            "request": request,
            "style": style,
            "style_guide": RULES.get(style, {}).get("스타일 가이드", ""),
            "rules": f'{RULES["공통"]}\n{RULES.get(style, {}).get("규칙", "")}\n관련 규칙:\n{formatted_rules}'
        })
        return _parameterize_template_internal(generated_template_text.strip())

    except Exception as e:
        print(f"Error in generate_template_old: {e}")
        return ParameterizedResult(parameterized_template="템플릿 생성 중 오류가 발생했습니다.", variables=[])


async def _convert_template_response_to_frontend_format(response_obj: TemplateResponse) -> dict:
    """TemplateResponse 객체를 프론트엔드가 기대하는 딕셔너리 형식으로 변환합니다."""
    structured_template = {
        "title": response_obj.template_data.title if hasattr(response_obj.template_data, 'title') else "",
        "body": response_obj.template_data.body if hasattr(response_obj.template_data, 'body') else "",
        "buttons": response_obj.template_data.buttons if hasattr(response_obj.template_data, 'buttons') else []
    }

    # editable_variables 추출 (예시, 실제 구현은 더 복잡할 수 있음)
    editable_variables = {}
    if hasattr(response_obj.template_data, 'body'):
        variables_in_body = re.findall(r'#\{\{(.*?)\}\}', response_obj.template_data.body)
        for var_name in variables_in_body:
            editable_variables[var_name] = ""
    if hasattr(response_obj.template_data, 'title'):
        variables_in_title = re.findall(r'#\{\{(.*?)\}\}', response_obj.template_data.title)
        for var_name in variables_in_title:
            editable_variables[var_name] = ""

    return {
        "structured_template": structured_template,
        "templates": [structured_template], # 'templates' 키 사용
        "template": response_obj.template_data.body, # 원본 템플릿 텍스트
        "editable_variables": editable_variables,
        "options": [], # 필요시 채움
        "hasImage": False # 이미지형 템플릿일 경우 True로 설정
    }


async def process_chat_message_async(user_message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """사용자 메시지를 처리하고 봇 응답을 생성합니다."""
    print(f"Received message: {user_message}")
    print(f"Current state: {state}")

    bot_response = {
        "content": "",
        "options": [],
        "template": None,
        "structured_template": None,
        "editable_variables": {},
        "templates": [],
        "hasImage": False,
        "intent": "",
        "next_action": ""
    }

    try:
        # 시스템 초기화 확인
        if llm_general is None:
            initialize_system()

        # 1. 의도 파악 (초기 요청 또는 상태에 따른 분기)
        if not state.get("intent"):
            # 초기 요청 처리: 템플릿 생성 의도 파악
            intent_prompt = ChatPromptTemplate.from_template(
                """사용자의 요청을 분석하여 다음 중 하나의 의도를 파악하세요: '템플릿_생성', '템플릿_수정', '템플릿_확인', '기타'.
                응답은 파악된 의도만 반환하세요. (예: 템플릿_생성)
                사용자 요청: {user_message}"""
            )
            intent_chain = intent_prompt | llm_fast | StrOutputParser()
            intent = await intent_chain.ainvoke({"user_message": user_message})
            state["intent"] = intent.strip()
            state["original_request"] = user_message # 최초 요청 저장

        current_intent = state.get("intent")
        next_action = state.get("next_action")

        if current_intent == "템플릿_생성":
            if not next_action:
                # 템플릿 스타일 선택 요청
                bot_response["content"] = "어떤 스타일의 템플릿을 생성해 드릴까요? (기본형, 이미지형, 강조형, 아이템 리스트형, 복합형)"
                bot_response["options"] = ["기본형", "이미지형", "강조형", "아이템 리스트형", "복합형"]
                state["next_action"] = "스타일_선택"

            elif next_action == "스타일_선택":
                selected_style = user_message.strip()
                if selected_style not in ["기본형", "이미지형", "강조형", "아이템 리스트형", "복합형"]:
                    bot_response["content"] = "죄송합니다. 지원하지 않는 스타일입니다. 다시 선택해주세요. (기본형, 이미지형, 강조형, 아이템 리스트형, 복합형)"
                    bot_response["options"] = ["기본형", "이미지형", "강조형", "아이템 리스트형", "복합형"]
                else:
                    state["selected_style"] = selected_style
                    bot_response["content"] = f"'{selected_style}' 스타일로 템플릿을 생성합니다. 잠시만 기다려주세요..."
                    
                    # 새로운 템플릿 생성 로직 호출
                    generated_template_obj = await generate_template_for_style(state["original_request"], selected_style)
                    
                    # TemplateResponse 객체 생성
                    final_template_response = TemplateResponse(
                        style=selected_style,
                        template_data=generated_template_obj
                    )
                    state["final_template_response"] = final_template_response.model_dump()

                    # 프론트엔드 형식으로 변환
                    frontend_format = await _convert_template_response_to_frontend_format(final_template_response)
                    bot_response.update(frontend_format)
                    bot_response["content"] = "템플릿 초안이 생성되었습니다. 수정할 부분이 있다면 말씀해주세요."
                    bot_response["options"] = ["수정하기", "완료"]
                    state["next_action"] = "템플릿_피드백"

            elif next_action == "템플릿_피드백":
                if user_message == "완료":
                    bot_response["content"] = "템플릿 생성이 완료되었습니다. 감사합니다!"
                    state.clear() # 상태 초기화
                elif user_message == "수정하기":
                    bot_response["content"] = "어떤 부분을 수정하고 싶으신가요? 구체적으로 말씀해주세요."
                    state["next_action"] = "템플릿_수정_요청"
                else:
                    # 피드백을 반영하여 템플릿 수정
                    bot_response["content"] = "피드백을 반영하여 템플릿을 수정 중입니다. 잠시만 기다려주세요..."
                    refined_template_dict = await refine_template_with_feedback_new(state, user_message)
                    
                    # 수정된 템플릿 객체로 업데이트
                    current_style = state["selected_style"]
                    style_configs = {
                        "기본형": BasicTemplate,
                        "이미지형": ImageTemplate,
                        "강조형": HighlightTemplate,
                        "아이템 리스트형": ItemListTemplate,
                        "복합형": CompositeTemplate
                    }
                    refined_template_obj = style_configs[current_style](**refined_template_dict)

                    final_template_response = TemplateResponse(
                        style=current_style,
                        template_data=refined_template_obj
                    )
                    state["final_template_response"] = final_template_response.model_dump()

                    # 프론트엔드 형식으로 변환
                    frontend_format = await _convert_template_response_to_frontend_format(final_template_response)
                    bot_response.update(frontend_format)
                    bot_response["content"] = "템플릿이 수정되었습니다. 추가 수정 사항이 있으신가요?"
                    bot_response["options"] = ["수정하기", "완료"]
                    state["next_action"] = "템플릿_피드백"

        elif current_intent == "템플릿_수정":
            # 기존 템플릿 수정 로직 (현재는 템플릿_생성 흐름에 통합)
            bot_response["content"] = "현재는 템플릿 생성 후 피드백을 통해 수정하는 방식만 지원합니다. 새로운 템플릿을 생성하시겠어요?"
            bot_response["options"] = ["네, 생성할게요", "아니요"]
            state.clear() # 상태 초기화

        elif current_intent == "템플릿_확인":
            # 템플릿 확인 로직 (구현 필요)
            bot_response["content"] = "어떤 템플릿을 확인하고 싶으신가요?"
            state["next_action"] = "템플릿_확인_요청"

        else:
            bot_response["content"] = "죄송합니다. 요청을 이해하지 못했습니다. 템플릿 생성에 대해 도와드릴 수 있습니다."
            state.clear() # 상태 초기화

        # 최종 상태 업데이트
        bot_response["intent"] = state.get("intent", "")
        bot_response["next_action"] = state.get("next_action", "")

    except Exception as e:
        print(f"Error in process_chat_message_async: {e}")
        traceback.print_exc()
        bot_response["content"] = f"처리 중 오류가 발생했습니다: {e}"
        bot_response["options"] = []
        state.clear() # 오류 발생 시 상태 초기화

    return bot_response


