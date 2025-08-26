# import os
# import json
# import re
# from typing import TypedDict, List, Optional
# from flask import Blueprint, request, jsonify, current_app
# from flask_cors import cross_origin
# import sys

# # 프로젝트 루트를 sys.path에 추가
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# sys.path.insert(0, project_root)

# # Pydantic 및 LangChain 호환성을 위한 임포트
# from pydantic import BaseModel, Field, PrivateAttr

# # LangChain 및 관련 라이브러리 임포트
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.document_loaders.base import BaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# from langchain_community.retrievers import BM25Retriever
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from langgraph.graph import StateGraph, END

# # FlashRank 임포트
# try:
#     from flashrank import Ranker, RerankRequest
#     from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
#     from langchain_core.callbacks.manager import Callbacks
# except ImportError:
#     print("FlashRank 또는 관련 모듈을 찾을 수 없습니다.")
#     BaseDocumentCompressor = object
#     Ranker = None

# chatbot_bp = Blueprint('chatbot', __name__)

# # 상수 정의
# MAX_CORRECTION_ATTEMPTS = 5

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
#     revised_template: Optional[str] = Field(None, description="수정된 템플릿")

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

# class GraphState(TypedDict):
#     original_request: str
#     user_choice: str
#     selected_style: str
#     template_draft: str
#     validation_result: Optional[TemplateAnalysisResult]
#     correction_attempts: int

# # 전역 변수들
# llm = None
# retrievers = {}
# approved_templates = []
# rejected_templates = []

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

# def initialize_system():
#     global llm, retrievers, approved_templates, rejected_templates
    
#     if llm is not None:
#         return  # 이미 초기화됨
    
#     # LLM 초기화
#     llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
#     # 데이터 로드
#     data_dir = os.path.join(project_root, 'data')
#     approved_templates = load_line_by_line(os.path.join(data_dir, "approved_templates.txt"))
#     rejected_templates = load_by_separator(os.path.join(data_dir, "rejected_templates.txt"))
    
#     # Retriever 설정
#     from chromadb.config import Settings
    
#     docs_compliance = CustomRuleLoader(os.path.join(data_dir, "compliance_rules.txt")).load()
#     docs_generation = CustomRuleLoader(os.path.join(data_dir, "generation_rules.txt")).load()
#     docs_whitelist = [Document(page_content=t) for t in approved_templates]
#     docs_rejected = [Document(page_content=t) for t in rejected_templates]
    
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     vector_db_path = os.path.join(project_root, "vector_db")
#     client_settings = Settings(anonymized_telemetry=False)
    
#     def create_db(name, docs):
#         if docs:
#             return Chroma.from_documents(
#                 docs, embeddings, 
#                 collection_name=name, 
#                 persist_directory=vector_db_path, 
#                 client_settings=client_settings
#             )
#         return Chroma(
#             collection_name=name, 
#             embedding_function=embeddings, 
#             persist_directory=vector_db_path, 
#             client_settings=client_settings
#         )
    
#     db_compliance = create_db("compliance_rules", docs_compliance)
#     db_generation = create_db("generation_rules", docs_generation)
#     db_whitelist = create_db("whitelist_templates", docs_whitelist)
#     db_rejected = create_db("rejected_templates", docs_rejected)
    
#     def create_hybrid_retriever(vectorstore, docs):
#         if not docs:
#             return vectorstore.as_retriever(search_kwargs={"k": 5})
        
#         vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#         keyword_retriever = BM25Retriever.from_documents(docs)
#         keyword_retriever.k = 5
        
#         ensemble_retriever = EnsembleRetriever(
#             retrievers=[vector_retriever, keyword_retriever], 
#             weights=[0.5, 0.5]
#         )
        
#         if Ranker:
#             compressor = FlashRankRerank(top_n=3)
#             return ContextualCompressionRetriever(
#                 base_compressor=compressor, 
#                 base_retriever=ensemble_retriever
#             )
#         return ensemble_retriever
    
#     retrievers['compliance'] = create_hybrid_retriever(db_compliance, docs_compliance)
#     retrievers['generation'] = create_hybrid_retriever(db_generation, docs_generation)
#     retrievers['whitelist'] = create_hybrid_retriever(db_whitelist, docs_whitelist)
#     retrievers['rejected'] = create_hybrid_retriever(db_rejected, docs_rejected)

# @chatbot_bp.route('/chat', methods=['POST'])
# @cross_origin()
# def chat():
#     try:
#         initialize_system()
        
#         data = request.get_json()
#         message = data.get('message', '')
#         session_state = data.get('state', {})
        
#         # 세션 상태 초기화
#         if not session_state:
#             session_state = {
#                 'step': 'initial',
#                 'original_request': '',
#                 'user_choice': '',
#                 'selected_style': '',
#                 'template_draft': '',
#                 'validation_result': None,
#                 'correction_attempts': 0
#             }
        
#         response = process_chat_message(message, session_state)
        
#         return jsonify({
#             'success': True,
#             'response': response['message'],
#             'state': response['state'],
#             'options': response.get('options', []),
#             'template': response.get('template', ''),
#             'step': response['state']['step']
#         })
        
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

# def process_chat_message(message: str, state: dict) -> dict:
#     """채팅 메시지 처리"""
    
#     if state['step'] == 'initial':
#         # 첫 요청 처리
#         state['original_request'] = message
#         state['step'] = 'recommend_templates'
        
#         # 유사 템플릿 검색
#         similar_docs = retrievers['whitelist'].invoke(message)
        
#         if not similar_docs:
#             state['step'] = 'select_style'
#             return {
#                 'message': '유사한 기존 템플릿을 찾지 못했습니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:',
#                 'state': state,
#                 'options': ['기본형', '이미지형', '아이템리스트형']
#             }
        
#         templates = []
#         for i, doc in enumerate(similar_docs[:3]):
#             templates.append(f"템플릿 {i+1}:\n{doc.page_content}")
        
#         return {
#             'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다:\n\n' + '\n\n'.join(templates) + '\n\n이 중에서 사용하실 템플릿을 선택하시거나, 새로운 템플릿 생성을 원하시면 "신규 생성"을 선택해주세요.',
#             'state': state,
#             'options': ['템플릿 1', '템플릿 2', '템플릿 3', '신규 생성'],
#             'templates': [doc.page_content for doc in similar_docs[:3]]
#         }
    
#     elif state['step'] == 'recommend_templates':
#         # 템플릿 선택 처리
#         if message in ['템플릿 1', '템플릿 2', '템플릿 3']:
#             template_idx = int(message.split()[1]) - 1
#             similar_docs = retrievers['whitelist'].invoke(state['original_request'])
#             state['template_draft'] = similar_docs[template_idx].page_content
#             state['step'] = 'validate'
            
#             # 검증 수행
#             validation_result = validate_template(state['template_draft'])
#             state['validation_result'] = validation_result
            
#             if validation_result['status'] == 'accepted':
#                 state['step'] = 'completed'
#                 return {
#                     'message': f'✅ 선택하신 템플릿이 규정을 준수합니다!\n\n최종 템플릿:\n{state["template_draft"]}',
#                     'state': state,
#                     'template': state['template_draft']
#                 }
#             else:
#                 state['step'] = 'correction'
#                 return {
#                     'message': f'🚨 선택하신 템플릿에 문제가 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result["suggestion"]}\n\nAI가 자동으로 수정하겠습니다.',
#                     'state': state
#                 }
        
#         elif message == '신규 생성':
#             state['step'] = 'select_style'
#             return {
#                 'message': '새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요:',
#                 'state': state,
#                 'options': ['기본형', '이미지형', '아이템리스트형']
#             }
    
#     elif state['step'] == 'select_style':
#         # 스타일 선택 처리
#         if message in ['기본형', '이미지형', '아이템리스트형']:
#             state['selected_style'] = message
#             state['step'] = 'generate'
            
#             # 템플릿 생성
#             template_draft = generate_template(state['original_request'], state['selected_style'])
#             state['template_draft'] = template_draft
            
#             # 검증 수행
#             validation_result = validate_template(template_draft)
#             state['validation_result'] = validation_result
            
#             if validation_result['status'] == 'accepted':
#                 state['step'] = 'completed'
#                 return {
#                     'message': f'✅ 생성된 템플릿이 규정을 준수합니다!\n\n최종 템플릿:\n{template_draft}',
#                     'state': state,
#                     'template': template_draft
#                 }
#             else:
#                 state['step'] = 'correction'
#                 state['correction_attempts'] = 0
#                 return {
#                     'message': f'템플릿을 생성했지만 규정 위반이 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result["suggestion"]}\n\nAI가 자동으로 수정하겠습니다.',
#                     'state': state
#                 }
    
#     elif state['step'] == 'correction':
#         # AI 자동 수정 수행
#         if state['correction_attempts'] < MAX_CORRECTION_ATTEMPTS:
#             corrected_template = correct_template(state)
#             state['template_draft'] = corrected_template
#             state['correction_attempts'] += 1
            
#             # 재검증
#             validation_result = validate_template(corrected_template)
#             state["validation_result"] = validation_result
            
#             if validation_result["status"] == "accepted":
#                 state["step"] = "completed"
#                 return {
#                     "message": f"✅ AI 수정이 완료되었습니다! 규정을 준수하는 템플릿이 생성되었습니다.\n\n최종 템플릿:\n{corrected_template}",
#                     "state": state,
#                     "template": corrected_template
#                 }
#             else:
#                 # AI가 수정했지만 여전히 문제가 있다면, 다시 correction 스텝으로 돌아가서 재시도
#                 return process_chat_message(corrected_template, state) # 재귀 호출로 자동 수정 반복
#         else:
#             # 최대 수정 횟수 초과
#             state['step'] = 'manual_correction'
#             return {
#                 'message': f'AI 자동 수정이 {MAX_CORRECTION_ATTEMPTS}회 모두 실패했습니다.\n\n현재 템플릿:\n{state["template_draft"]}\n\n마지막 문제점: {state["validation_result"]["reason"]}\n\n직접 수정하시겠습니까? 수정할 내용을 입력해주세요.',
#                 'state': state,
#                 'options': ['포기하기']
#             }
    
#     elif state['step'] == 'manual_correction':
#         if message == '포기하기':
#             state['step'] = 'initial'
#             return {
#                 'message': '템플릿 생성을 포기했습니다. 새로운 요청을 입력해주세요.',
#                 'state': {'step': 'initial'},
#             }
#         else:
#             # 사용자가 직접 수정한 템플릿
#             state['template_draft'] = message
            
#             # 최종 검증
#             validation_result = validate_template(message)
#             state['validation_result'] = validation_result
            
#             if validation_result['status'] == 'accepted':
#                 state['step'] = 'completed'
#                 return {
#                     'message': f'✅ 사용자 수정이 완료되었습니다! 규정을 준수하는 템플릿이 생성되었습니다.\n\n최종 템플릿:\n{message}',
#                     'state': state,
#                     'template': message
#                 }
#             else:
#                 return {
#                     'message': f'🚨 수정하신 템플릿에도 여전히 문제가 있습니다.\n\n문제점: {validation_result["reason"]}\n\n다시 수정해주시거나 "포기하기"를 선택해주세요.',
#                     'state': state,
#                     'options': ['포기하기']
#                 }
    
#     elif state['step'] == 'completed':
#         # 새로운 요청 시작
#         state = {'step': 'initial'}
#         return process_chat_message(message, state)
    
#     return {
#         'message': '죄송합니다. 처리할 수 없는 요청입니다.',
#         'state': state
#     }

# def generate_template(request: str, style: str) -> str:
#     """템플릿 생성"""
#     example_docs = retrievers['whitelist'].invoke(request)
#     examples = "\n\n".join([f"예시 {i+1}:\n{doc.page_content}" for i, doc in enumerate(example_docs)])
    
#     expansion_prompt = ChatPromptTemplate.from_template(
#         """당신은 사용자의 핵심 의도와 '선택된 스타일'을 바탕으로, 정보가 풍부한 알림톡 템플릿 초안을 확장하는 전문가입니다.
#         당신의 유일한 임무는 아래 지시사항에 따라 **정보가 확장된 템플릿 초안 하나만**을 생성하는 것입니다. 초안 외에 다른 설명은 절대로 덧붙이지 마세요.
        
#         # 지시사항
#         1. '사용자 핵심 의도'를 바탕으로, '선택된 스타일'에 맞는 완전한 템플릿 초안을 만드세요.
#         2. '유사한 성공 사례'를 참고하여, 어떤 정보(예: 지원 대상, 신청 기간 등)를 추가해야 할지 **추론**하고, 적절한 #{{변수}}를 사용하세요.
        
#         # 사용자 핵심 의도: {original_request}
#         # 선택된 스타일: {style}
#         # 유사한 성공 사례 (참고용): {examples}
        
#         # 확장된 템플릿 초안 (오직 템플릿 텍스트만 출력):"""
#     )
    
#     expansion_chain = expansion_prompt | llm | StrOutputParser()
#     expanded_draft = expansion_chain.invoke({
#         "original_request": request,
#         "style": style,
#         "examples": examples
#     })
    
#     return expanded_draft

# def validate_template(draft: str) -> dict:
#     """템플릿 검증"""
#     parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
    
#     step_back_chain = ChatPromptTemplate.from_template(
#         "이 템플릿의 핵심 쟁점은 무엇인가?: {draft}"
#     ) | llm | StrOutputParser()
    
#     step_back_question = step_back_chain.invoke({"draft": draft})
    
#     compliance_docs = retrievers['compliance'].invoke(step_back_question)
#     rules_with_metadata = "\n\n".join([
#         f"문서 내용: {doc.page_content}" 
#         for doc in compliance_docs
#     ])
    
#     rejected_docs = retrievers['rejected'].invoke(draft)
#     rejections = "\n\n".join([doc.page_content for doc in rejected_docs])
    
#     validation_prompt = ChatPromptTemplate.from_template(
#         """당신은 과거 판례와 법규를 근거로 판단하는 매우 꼼꼼한 최종 심사관입니다.
#         주어진 JSON 형식에 맞춰서만 답변해야 합니다.

#         # 검수 대상 템플릿: {draft}
#         # 관련 규정 (메타데이터 포함): {rules}
#         # 유사한 과거 반려 사례 (판례): {rejections}

#         # 지시사항
#         1. 'reason' 필드를 다음과 같은 다단계 추론 과정에 따라 상세하게 작성하세요:
#             a. **사실 확인:** 먼저, '검수 대상 템플릿'에 어떤 내용이 포함되어 있는지 객관적으로 서술하세요.
#             b. **규정 연결:** 다음으로, 확인된 사실과 가장 관련성이 높은 '관련 규정' 또는 '유사한 과거 반려 사례'를 1~3개 찾아 연결하세요.
#             c. **최종 결론:** 마지막으로, 위 사실과 규정을 종합하여 왜 이 템플릿이 'accepted' 또는 'rejected'인지 명확한 결론을 내리세요.
#         2. 위반 사항이 없다면 'status'를 'accepted'로 설정하고, 'revised_template'에 원본 초안을 그대로 넣으세요.
#         3. 위반 사항이 있다면 'status'를 'rejected'로 설정하고, 'suggestion'에 구체적인 개선 방안을 제시하세요.

#         # 출력 형식 (JSON):
#         {format_instructions}
#         """
#     )
    
#     validation_chain = validation_prompt | llm | parser
#     result = validation_chain.invoke({
#         "draft": draft,
#         "rules": rules_with_metadata,
#         "rejections": rejections,
#         "format_instructions": parser.get_format_instructions()
#     })
    
#     return result

# def correct_template(state: dict) -> str:
#     """템플릿 수정"""
#     attempts = state.get('correction_attempts', 0)
    
#     if attempts == 0:
#         instruction = "3. 광고성 문구를 제거하거나, 정보성 내용으로 순화하는 등, 제안된 방향에 맞게 템플릿을 수정하세요."
#     elif attempts == 1:
#         instruction = "3. **(2차 수정)** 아직도 문제가 있습니다. 이번에는 '쿠폰', '할인', '이벤트', '특가'와 같은 명백한 광고성 단어를 사용하지 마세요."
#     else:
#         instruction = """3. **(최종 수정: 관점 전환)** 여전히 광고성으로 보입니다. 이것이 마지막 시도입니다.
#         - **관점 전환:** 메시지의 주체를 '우리(사업자)'에서 '고객님'으로 완전히 바꾸세요.
#         - **목적 변경:** '판매'나 '방문 유도'가 아니라, '고객님이 과거에 동의한 내용에 따라 고객님의 권리(혜택) 정보를 안내'하는 것으로 목적을 재정의하세요."""
    
#     correction_prompt_template = """당신은 지적된 문제점을 해결하여 더 나은 대안을 제시하는 전문 카피라이터입니다.
#     당신의 유일한 임무는 아래 지시사항에 따라 **수정된 템플릿 초안 하나만**을 생성하는 것입니다. 초안 외에 다른 설명은 절대로 덧붙이지 마세요.

#     # 원래 사용자 요청: {original_request}
#     # 이전에 제안했던 템플릿 (반려됨): {rejected_draft}
#     # 반려 사유 및 개선 제안: {rejection_reason}

#     # 지시사항
#     1. '반려 사유 및 개선 제안'을 완벽하게 이해하고, 지적된 모든 문제점을 해결하세요.
#     2. '원래 사용자 요청'의 핵심 의도는 유지해야 합니다.
#     {dynamic_instruction}

#     # 수정된 템플릿 초안 (오직 템플릿 텍스트만 출력):
#     """
    
#     correction_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
#     correction_prompt = correction_prompt.partial(dynamic_instruction=instruction)
    
#     correction_chain = correction_prompt | llm | StrOutputParser()
    
#     new_draft = correction_chain.invoke({
#         "original_request": state['original_request'],
#         "rejected_draft": state['template_draft'],
#         "rejection_reason": state['validation_result']['reason'] + "\n개선 제안: " + state['validation_result']['suggestion']
#     })
    
#     return new_draft

# @chatbot_bp.route('/health', methods=['GET'])
# def health():
#     return jsonify({'status': 'healthy'})

import os
import json
import re
from typing import TypedDict, List, Optional, Dict
from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Pydantic 및 LangChain 호환성을 위한 임포트
from pydantic import BaseModel, Field, PrivateAttr

# LangChain 및 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END

# FlashRank 임포트
try:
    from flashrank import Ranker, RerankRequest
    from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
    from langchain_core.callbacks.manager import Callbacks
except ImportError:
    print("FlashRank 또는 관련 모듈을 찾을 수 없습니다.")
    BaseDocumentCompressor = object
    Ranker = None

chatbot_bp = Blueprint('chatbot', __name__)

# 상수 정의
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
    revised_template: Optional[str] = Field(None, description="수정된 템플릿")

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

class GraphState(TypedDict):
    original_request: str
    user_choice: str
    selected_style: str
    template_draft: str
    validation_result: Optional[TemplateAnalysisResult]
    correction_attempts: int
    
# --- [신규 추가] 변수 추출을 위한 Pydantic 모델 ---
class Variable(BaseModel):
    name: str = Field(description="추출된 변수의 한글 이름 (예: 매장명, 폐점일자). `#{}`에 들어갈 부분입니다.")
    original_value: str = Field(description="원본 텍스트에서 추출된 실제 값")
    description: str = Field(description="해당 변수에 대한 간단한 한글 설명 (사용자가 이해하기 쉽도록)")

class ParameterizedResult(BaseModel):
    parameterized_template: str = Field(description="특정 정보가 #{변수명}으로 대체된 최종 템플릿")
    variables: List[Variable] = Field(description="추출된 변수들의 목록")


# 전역 변수들
llm = None
retrievers = {}
approved_templates = []
rejected_templates = []

# --- [신규 추가] HTML 미리보기 생성 함수 ---
def render_final_template(template_string: str) -> str:
    """최종 승인된 템플릿 문자열을 받아 이미지와 유사한 HTML 미리보기를 생성합니다."""
    
    lines = [line.strip() for line in template_string.strip().split('\n') if line.strip()]
    
    if not lines:
        return "<span>미리보기를 생성할 템플릿이 없습니다.</span>"

    title = lines[0]
    button_text = lines[-1]
    
    body_lines = []
    note_lines = []
    for line in lines[1:-1]:
        if line.startswith('*'):
            note_lines.append(line)
        else:
            body_lines.append(line)
            
    body = "<br>".join(body_lines)
    note = "<br>".join(note_lines)

    body = re.sub(r'(#{\w+})', r'<span class="placeholder">\1</span>', body)

    html_output = f"""
    <div class="template-preview">
        <div class="header">알림톡 도착</div>
        <div class="content">
            <div class="icon">📄</div>
            <h2 class="title">{title}</h2>
            <div class="body-text">
                {body}
            </div>
            <div class="note-text">
                {note}
            </div>
            <div class="button-container">
                <span>{button_text}</span>
            </div>
        </div>
    </div>
    <style>
        .template-preview {{
            max-width: 350px; border-radius: 8px; overflow: hidden;
            font-family: 'Malgun Gothic', '맑은 고딕', sans-serif;
            border: 1px solid #e0e0e0; margin: 1em 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .template-preview .header {{
            background-color: #F0CA4F; color: #333; padding: 10px 15px;
            font-weight: bold; font-size: 14px;
        }}
        .template-preview .content {{
            background-color: #E6ECF2; padding: 25px 20px; position: relative;
        }}
        .template-preview .icon {{
            position: absolute; top: 25px; right: 20px; font-size: 36px; opacity: 0.5;
        }}
        .template-preview .title {{
            font-size: 24px; font-weight: bold; margin: 0 0 20px;
            padding-right: 40px; color: #333;
        }}
        .template-preview .body-text {{
            font-size: 15px; line-height: 1.6; color: #555; margin-bottom: 20px;
        }}
        .template-preview .note-text {{
            font-size: 13px; line-height: 1.5; color: #777; margin-bottom: 25px;
        }}
        .template-preview .placeholder {{ color: #007bff; font-weight: bold; }}
        .template-preview .button-container {{
            background-color: #FFFFFF; border: 1px solid #d0d0d0; border-radius: 5px;
            text-align: center; padding: 12px 10px; font-size: 15px;
            font-weight: bold; color: #007bff; cursor: pointer;
        }}
    </style>
    """
    return html_output

# --- [신규 추가] 변수 자동 추출 함수 ---
def parameterize_template(template_string: str) -> Dict:
    """완성된 템플릿 문자열을 받아 주요 정보를 변수화합니다."""
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
        return result
    except Exception as e:
        print(f"Error during parameterization: {e}")
        return {"parameterized_template": template_string, "variables": []}


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

def initialize_system():
    global llm, retrievers, approved_templates, rejected_templates
    
    if llm is not None:
        return  # 이미 초기화됨
    
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # 데이터 로드
    data_dir = os.path.join(project_root, 'data')
    approved_templates = load_line_by_line(os.path.join(data_dir, "approved_templates.txt"))
    rejected_templates = load_by_separator(os.path.join(data_dir, "rejected_templates.txt"))
    
    # Retriever 설정
    from chromadb.config import Settings
    
    docs_compliance = CustomRuleLoader(os.path.join(data_dir, "compliance_rules.txt")).load()
    docs_generation = CustomRuleLoader(os.path.join(data_dir, "generation_rules.txt")).load()
    docs_whitelist = [Document(page_content=t) for t in approved_templates]
    docs_rejected = [Document(page_content=t) for t in rejected_templates]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db_path = os.path.join(project_root, "vector_db")
    client_settings = Settings(anonymized_telemetry=False)
    
    def create_db(name, docs):
        if docs:
            return Chroma.from_documents(
                docs, embeddings, 
                collection_name=name, 
                persist_directory=vector_db_path, 
                client_settings=client_settings
            )
        return Chroma(
            collection_name=name, 
            embedding_function=embeddings, 
            persist_directory=vector_db_path, 
            client_settings=client_settings
        )
    
    db_compliance = create_db("compliance_rules", docs_compliance)
    db_generation = create_db("generation_rules", docs_generation)
    db_whitelist = create_db("whitelist_templates", docs_whitelist)
    db_rejected = create_db("rejected_templates", docs_rejected)
    
    def create_hybrid_retriever(vectorstore, docs):
        if not docs:
            return vectorstore.as_retriever(search_kwargs={"k": 5})
        
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever.k = 5
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever], 
            weights=[0.5, 0.5]
        )
        
        if Ranker:
            compressor = FlashRankRerank(top_n=3)
            return ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=ensemble_retriever
            )
        return ensemble_retriever
    
    retrievers['compliance'] = create_hybrid_retriever(db_compliance, docs_compliance)
    retrievers['generation'] = create_hybrid_retriever(db_generation, docs_generation)
    retrievers['whitelist'] = create_hybrid_retriever(db_whitelist, docs_whitelist)
    retrievers['rejected'] = create_hybrid_retriever(db_rejected, docs_rejected)

@chatbot_bp.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    try:
        initialize_system()
        
        data = request.get_json()
        message = data.get('message', '')
        session_state = data.get('state', {})
        
        # 세션 상태 초기화
        if not session_state:
            session_state = {
                'step': 'initial',
                'original_request': '',
                'user_choice': '',
                'selected_style': '',
                'template_draft': '',
                'validation_result': None,
                'correction_attempts': 0
            }
        
        response = process_chat_message(message, session_state)
        
        return jsonify({
            'success': True,
            'response': response['message'],
            'state': response['state'],
            'options': response.get('options', []),
            'template': response.get('template', ''),
            'html_preview': response.get('html_preview', ''),
            'editable_variables': response.get('editable_variables', {}),
            'step': response['state']['step']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def process_chat_message(message: str, state: dict) -> dict:
    """채팅 메시지 처리"""
    
    if state['step'] == 'initial':
        # 첫 요청 처리
        state['original_request'] = message
        state['step'] = 'recommend_templates'
        
        # 유사 템플릿 검색
        similar_docs = retrievers['whitelist'].invoke(message)
        
        if not similar_docs:
            state['step'] = 'select_style'
            return {
                'message': '유사한 기존 템플릿을 찾지 못했습니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:',
                'state': state,
                'options': ['기본형', '이미지형', '아이템리스트형']
            }
        
        templates = []
        for i, doc in enumerate(similar_docs[:3]):
            templates.append(f"템플릿 {i+1}:\n{doc.page_content}")
        
        return {
            'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다:\n\n' + '\n\n'.join(templates) + '\n\n이 중에서 사용하실 템플릿을 선택하시거나, 새로운 템플릿 생성을 원하시면 "신규 생성"을 선택해주세요.',
            'state': state,
            'options': ['템플릿 1', '템플릿 2', '템플릿 3', '신규 생성'],
            'templates': [doc.page_content for doc in similar_docs[:3]]
        }
    
    elif state['step'] == 'recommend_templates':
        # 템플릿 선택 처리
        if message in ['템플릿 1', '템플릿 2', '템플릿 3']:
            template_idx = int(message.split()[1]) - 1
            similar_docs = retrievers['whitelist'].invoke(state['original_request'])
            state['template_draft'] = similar_docs[template_idx].page_content
            state['step'] = 'validate'
            
            # 검증 수행
            validation_result = validate_template(state['template_draft'])
            state['validation_result'] = validation_result
            
            if validation_result['status'] == 'accepted':
                state['step'] = 'completed'
                final_template = state["template_draft"]
                html_preview = render_final_template(final_template)
                parameterized_result = parameterize_template(final_template)
                return {
                    'message': f'✅ 선택하신 템플릿이 규정을 준수합니다!\n\n최종 템플릿:\n{final_template}',
                    'state': state,
                    'template': final_template,
                    'html_preview': html_preview,
                    'editable_variables': parameterized_result
                }
            else:
                state['step'] = 'correction'
                return {
                    'message': f'🚨 선택하신 템플릿에 문제가 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result["suggestion"]}\n\nAI가 자동으로 수정하겠습니다.',
                    'state': state
                }
        
        elif message == '신규 생성':
            state['step'] = 'select_style'
            return {
                'message': '새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요:',
                'state': state,
                'options': ['기본형', '이미지형', '아이템리스트형']
            }
    
    elif state['step'] == 'select_style':
        # 스타일 선택 처리
        if message in ['기본형', '이미지형', '아이템리스트형']:
            state['selected_style'] = message
            state['step'] = 'generate'
            
            # 템플릿 생성
            template_draft = generate_template(state['original_request'], state['selected_style'])
            state['template_draft'] = template_draft
            
            # 검증 수행
            validation_result = validate_template(template_draft)
            state['validation_result'] = validation_result
            
            if validation_result['status'] == 'accepted':
                state['step'] = 'completed'
                final_template = template_draft
                html_preview = render_final_template(final_template)
                parameterized_result = parameterize_template(final_template)
                return {
                    'message': f'✅ 생성된 템플릿이 규정을 준수합니다!\n\n최종 템플릿:\n{final_template}',
                    'state': state,
                    'template': final_template,
                    'html_preview': html_preview,
                    'editable_variables': parameterized_result
                }
            else:
                state['step'] = 'correction'
                state['correction_attempts'] = 0
                return process_chat_message('', state) # 자동 수정 시작
    
    elif state['step'] == 'correction':
        # AI 자동 수정 수행
        if state['correction_attempts'] < MAX_CORRECTION_ATTEMPTS:
            corrected_template = correct_template(state)
            state['template_draft'] = corrected_template
            state['correction_attempts'] += 1
            
            # 재검증
            validation_result = validate_template(corrected_template)
            state["validation_result"] = validation_result
            
            if validation_result["status"] == "accepted":
                state["step"] = "completed"
                final_template = corrected_template
                html_preview = render_final_template(final_template)
                parameterized_result = parameterize_template(final_template)
                return {
                    "message": f"✅ AI 수정이 완료되었습니다! 규정을 준수하는 템플릿이 생성되었습니다.\n\n최종 템플릿:\n{final_template}",
                    "state": state,
                    "template": final_template,
                    'html_preview': html_preview,
                    'editable_variables': parameterized_result
                }
            else:
                # 수정 후에도 실패하면 재귀 호출로 자동 수정 반복
                return process_chat_message(corrected_template, state) 
        else:
            # 최대 수정 횟수 초과
            state['step'] = 'manual_correction'
            return {
                'message': f'AI 자동 수정이 {MAX_CORRECTION_ATTEMPTS}회 모두 실패했습니다.\n\n현재 템플릿:\n{state["template_draft"]}\n\n마지막 문제점: {state["validation_result"]["reason"]}\n\n직접 수정하시겠습니까? 수정할 내용을 입력해주세요.',
                'state': state,
                'options': ['포기하기']
            }
    
    elif state['step'] == 'manual_correction':
        if message == '포기하기':
            state['step'] = 'initial'
            return {
                'message': '템플릿 생성을 포기했습니다. 새로운 요청을 입력해주세요.',
                'state': {'step': 'initial'},
            }
        else:
            # 사용자가 직접 수정한 템플릿
            state['template_draft'] = message
            
            # 최종 검증
            validation_result = validate_template(message)
            state['validation_result'] = validation_result
            
            if validation_result['status'] == 'accepted':
                state['step'] = 'completed'
                final_template = message
                html_preview = render_final_template(final_template)
                parameterized_result = parameterize_template(final_template)
                return {
                    'message': f'✅ 사용자 수정이 완료되었습니다! 규정을 준수하는 템플릿이 생성되었습니다.\n\n최종 템플릿:\n{final_template}',
                    'state': state,
                    'template': final_template,
                    'html_preview': html_preview,
                    'editable_variables': parameterized_result
                }
            else:
                return {
                    'message': f'🚨 수정하신 템플릿에도 여전히 문제가 있습니다.\n\n문제점: {validation_result["reason"]}\n\n다시 수정해주시거나 "포기하기"를 선택해주세요.',
                    'state': state,
                    'options': ['포기하기']
                }
    
    elif state['step'] == 'completed':
        # 새로운 요청 시작
        state = {'step': 'initial'}
        return process_chat_message(message, state)
    
    return {
        'message': '죄송합니다. 처리할 수 없는 요청입니다.',
        'state': state
    }

def generate_template(request: str, style: str) -> str:
    """템플릿 생성"""
    example_docs = retrievers['whitelist'].invoke(request)
    examples = "\n\n".join([f"예시 {i+1}:\n{doc.page_content}" for i, doc in enumerate(example_docs)])
    
    expansion_prompt = ChatPromptTemplate.from_template(
        """당신은 사용자의 핵심 의도와 '선택된 스타일'을 바탕으로, 정보가 풍부한 알림톡 템플릿 초안을 확장하는 전문가입니다.
        당신의 유일한 임무는 아래 지시사항에 따라 **정보가 확장된 템플릿 초안 하나만**을 생성하는 것입니다. 초안 외에 다른 설명은 절대로 덧붙이지 마세요.
        
        # 지시사항
        1. '사용자 핵심 의도'를 바탕으로, '선택된 스타일'에 맞는 완전한 템플릿 초안을 만드세요.
        2. '유사한 성공 사례'를 참고하여, 어떤 정보(예: 지원 대상, 신청 기간 등)를 추가해야 할지 **추론**하고, 적절한 #{{변수}}를 사용하세요.
        
        # 사용자 핵심 의도: {original_request}
        # 선택된 스타일: {style}
        # 유사한 성공 사례 (참고용): {examples}
        
        # 확장된 템플릿 초안 (오직 템플릿 텍스트만 출력):"""
    )
    
    expansion_chain = expansion_prompt | llm | StrOutputParser()
    expanded_draft = expansion_chain.invoke({
        "original_request": request,
        "style": style,
        "examples": examples
    })
    
    return expanded_draft

def validate_template(draft: str) -> dict:
    """템플릿 검증"""
    parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
    
    step_back_chain = ChatPromptTemplate.from_template(
        "이 템플릿의 핵심 쟁점은 무엇인가?: {draft}"
    ) | llm | StrOutputParser()
    
    step_back_question = step_back_chain.invoke({"draft": draft})
    
    compliance_docs = retrievers['compliance'].invoke(step_back_question)
    rules_with_metadata = "\n\n".join([
        f"문서 내용: {doc.page_content}" 
        for doc in compliance_docs
    ])
    
    rejected_docs = retrievers['rejected'].invoke(draft)
    rejections = "\n\n".join([doc.page_content for doc in rejected_docs])
    
    validation_prompt = ChatPromptTemplate.from_template(
        """당신은 과거 판례와 법규를 근거로 판단하는 매우 꼼꼼한 최종 심사관입니다.
        주어진 JSON 형식에 맞춰서만 답변해야 합니다.

        # 검수 대상 템플릿: {draft}
        # 관련 규정 (메타데이터 포함): {rules}
        # 유사한 과거 반려 사례 (판례): {rejections}

        # 지시사항
        1. 'reason' 필드를 다음과 같은 다단계 추론 과정에 따라 상세하게 작성하세요:
            a. **사실 확인:** 먼저, '검수 대상 템플릿'에 어떤 내용이 포함되어 있는지 객관적으로 서술하세요.
            b. **규정 연결:** 다음으로, 확인된 사실과 가장 관련성이 높은 '관련 규정' 또는 '유사한 과거 반려 사례'를 1~3개 찾아 연결하세요.
            c. **최종 결론:** 마지막으로, 위 사실과 규정을 종합하여 왜 이 템플릿이 'accepted' 또는 'rejected'인지 명확한 결론을 내리세요.
        2. 위반 사항이 없다면 'status'를 'accepted'로 설정하고, 'revised_template'에 원본 초안을 그대로 넣으세요.
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
    
    return result

def correct_template(state: dict) -> str:
    """템플릿 수정"""
    attempts = state.get('correction_attempts', 0)
    
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
    
    new_draft = correction_chain.invoke({
        "original_request": state['original_request'],
        "rejected_draft": state['template_draft'],
        "rejection_reason": state['validation_result']['reason'] + "\n개선 제안: " + state['validation_result']['suggestion']
    })
    
    return new_draft

@chatbot_bp.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})