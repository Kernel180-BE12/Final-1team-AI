# import os
# import json
# import re
# from typing import TypedDict, List, Optional, Dict
# import sys

# # Pydantic 및 LangChain 호환성을 위한 임포트
# from pydantic import BaseModel, Field, PrivateAttr

# # LangChain 및 관련 라이브러리 임포트
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.document_loaders.base import BaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# from langchain_community.retrievers import BM25Retriever
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# # FlashRank 임포트
# try:
#     from flashrank import Ranker, RerankRequest
#     from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
#     from langchain_core.callbacks.manager import Callbacks
# except ImportError:
#     print("FlashRank 또는 관련 모듈을 찾을 수 없습니다.")
#     BaseDocumentCompressor = object
#     Ranker = None

# # --- 설정 및 모델 정의 ---
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

# MAX_CORRECTION_ATTEMPTS = 3

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

# class Variable(BaseModel):
#     name: str = Field(description="추출된 변수의 한글 이름 (예: 매장명, 폐점일자). `#{}`에 들어갈 부분입니다.")
#     original_value: str = Field(description="원본 텍스트에서 추출된 실제 값")
#     description: str = Field(description="해당 변수에 대한 간단한 한글 설명 (사용자가 이해하기 쉽도록)")

# class ParameterizedResult(BaseModel):
#     parameterized_template: str = Field(description="특정 정보가 #{변수명}으로 대체된 최종 템플릿")
#     variables: List[Variable] = Field(description="추출된 변수들의 목록")

# # --- 전역 변수 및 헬퍼 함수 ---
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

# def render_final_template(template_string: str) -> str:
#     lines = [line.strip() for line in template_string.strip().split('\n') if line.strip()]
#     if not lines:
#         return "<span>미리보기를 생성할 템플릿이 없습니다.</span>"
#     title = lines[0]
#     button_text = lines[-1]
#     body_lines = []
#     note_lines = []
#     for line in lines[1:-1]:
#         if line.startswith('*'):
#             note_lines.append(line)
#         else:
#             body_lines.append(line)
#     body = "<br>".join(body_lines)
#     note = "<br>".join(note_lines)
#     body = re.sub(r'(#{\w+})', r'<span class="placeholder">\1</span>', body)
#     html_output = f"""
#     <div class="template-preview">
#         <div class="header">알림톡 도착</div>
#         <div class="content">
#             <div class="icon">📄</div>
#             <h2 class="title">{title}</h2>
#             <div class="body-text">{body}</div>
#             <div class="note-text">{note}</div>
#             <div class="button-container"><span>{button_text}</span></div>
#         </div>
#     </div>
#     <style>
#         .template-preview {{ max-width: 350px; border-radius: 8px; overflow: hidden; font-family: 'Malgun Gothic', '맑은 고딕', sans-serif; border: 1px solid #e0e0e0; margin: 1em 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
#         .template-preview .header {{ background-color: #F0CA4F; color: #333; padding: 10px 15px; font-weight: bold; font-size: 14px; }}
#         .template-preview .content {{ background-color: #E6ECF2; padding: 25px 20px; position: relative; }}
#         .template-preview .icon {{ position: absolute; top: 25px; right: 20px; font-size: 36px; opacity: 0.5; }}
#         .template-preview .title {{ font-size: 24px; font-weight: bold; margin: 0 0 20px; padding-right: 40px; color: #333; }}
#         .template-preview .body-text {{ font-size: 15px; line-height: 1.6; color: #555; margin-bottom: 20px; }}
#         .template-preview .note-text {{ font-size: 13px; line-height: 1.5; color: #777; margin-bottom: 25px; }}
#         .template-preview .placeholder {{ color: #007bff; font-weight: bold; }}
#         .template-preview .button-container {{ background-color: #FFFFFF; border: 1px solid #d0d0d0; border-radius: 5px; text-align: center; padding: 12px 10px; font-size: 15px; font-weight: bold; color: #007bff; cursor: pointer; }}
#     </style>
#     """
#     return html_output

# def parameterize_template(template_string: str) -> Dict:
#     parser = JsonOutputParser(pydantic_object=ParameterizedResult)
#     prompt = ChatPromptTemplate.from_template(
#         """당신은 주어진 텍스트를 재사용 가능한 템플릿으로 변환하는 전문가입니다.
#         주어진 텍스트에서 고유명사, 날짜, 장소, 숫자 등 구체적이고 바뀔 수 있는 정보들을 식별하여, 의미 있는 한글 변수명으로 대체해주세요.
#         # 지시사항
#         1. 텍스트의 핵심 정보(누가, 언제, 어디서, 무엇을, 어떻게 등)를 파악합니다.
#         2. 파악된 정보를 `#{{변수명}}` 형태로 대체하여 재사용 가능한 템플릿을 생성합니다. 변수명은 사용자가 이해하기 쉬운 한글로 작성하세요.
#         3. 원본 값과 변수명, 그리고 각 변수에 대한 설명을 포함하는 변수 목록을 생성합니다.
#         4. 최종 결과를 지정된 JSON 형식으로만 출력해야 합니다. 그 외의 설명은 절대 추가하지 마세요.
#         # 원본 텍스트:
#         {original_text}
#         # 출력 형식 (JSON):
#         {format_instructions}
#         """
#     )
#     chain = prompt | llm | parser
#     try:
#         result = chain.invoke({
#             "original_text": template_string,
#             "format_instructions": parser.get_format_instructions(),
#         })
#         return result
#     except Exception as e:
#         print(f"Error during parameterization: {e}")
#         return {"parameterized_template": template_string, "variables": []}

# def initialize_system():
#     global llm, retrievers, approved_templates, rejected_templates
#     if llm is not None:
#         return
#     llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
#     data_dir = os.path.join(project_root, 'data')
#     approved_templates = load_line_by_line(os.path.join(data_dir, "approved_templates.txt"))
#     rejected_templates = load_by_separator(os.path.join(data_dir, "rejected_templates.txt"))
#     from chromadb.config import Settings
#     docs_compliance = CustomRuleLoader(os.path.join(data_dir, "compliance_rules.txt")).load()
#     docs_generation = CustomRuleLoader(os.path.join(data_dir, "generation_rules.txt")).load()
#     docs_whitelist = [Document(page_content=t) for t in approved_templates]
#     docs_rejected = [Document(page_content=t) for t in rejected_templates]
#     """
#         추후에 반려된 템플릿 추가하면 고도화 가능
#     """
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     vector_db_path = os.path.join(project_root, "vector_db")
#     client_settings = Settings(anonymized_telemetry=False)
#     def create_db(name, docs):
#         if docs:
#             return Chroma.from_documents(docs, embeddings, collection_name=name, persist_directory=vector_db_path, client_settings=client_settings)
#         """
#             아래 return Chroma (~~)는 data 에 없는데 사용하고싶은 백터 db 가 있다면 사용.
#             불필요한 코드.
#         """
#         # return Chroma(collection_name=name, embedding_function=embeddings, persist_directory=vector_db_path, client_settings=client_settings)
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
#         ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])
#         if Ranker:
#             compressor = FlashRankRerank(top_n=5)
#             return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
#         return ensemble_retriever
#     retrievers['compliance'] = create_hybrid_retriever(db_compliance, docs_compliance)
#     retrievers['generation'] = create_hybrid_retriever(db_generation, docs_generation)
#     retrievers['whitelist'] = create_hybrid_retriever(db_whitelist, docs_whitelist)
#     retrievers['rejected'] = create_hybrid_retriever(db_rejected, docs_rejected)

# # ★★★ [수정] process_chat_message 함수 전체를 교체해주세요. ★★★
# def process_chat_message(message: str, state: dict) -> dict:
#     """채팅 메시지 처리"""
    
#     if state['step'] == 'initial':
#         state['original_request'] = message
#         state['step'] = 'recommend_templates'
#         similar_docs = retrievers['whitelist'].invoke(message)
        
#         if not similar_docs:
#             state['step'] = 'select_style'
#             return {
#                 'message': '유사한 기존 템플릿을 찾지 못했습니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:',
#                 'state': state,
#                 'options': ['기본형', '이미지형', '아이템리스트형']
#             }
        
#         templates = [f"템플릿 {i+1}:\n{doc.page_content}" for i, doc in enumerate(similar_docs[:3])]
#         return {
#             'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다:\n\n' + '\n\n'.join(templates) + '\n\n이 중에서 사용하실 템플릿을 선택하시거나, 새로운 템플릿 생성을 원하시면 "신규 생성"을 선택해주세요.',
#             'state': state,
#             'options': ['템플릿 1', '템플릿 2', '템플릿 3', '신규 생성'],
#             'templates': [doc.page_content for doc in similar_docs[:3]]
#         }
    
#     elif state['step'] == 'recommend_templates':
#         if message in ['템플릿 1', '템플릿 2', '템플릿 3']:
#             template_idx = int(message.split()[1]) - 1
#             similar_docs = retrievers['whitelist'].invoke(state['original_request'])
#             state['template_draft'] = similar_docs[template_idx].page_content
#         elif message == '신규 생성':
#             state['step'] = 'select_style'
#             return {
#                 'message': '새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요:',
#                 'state': state,
#                 'options': ['기본형', '이미지형', '아이템리스트형']
#             }
#         else: # 사용자가 직접 텍스트를 입력한 경우 (스타일 선택으로 바로 넘어감)
#              state['step'] = 'select_style'
#              return process_chat_message(message, state)

#     if state.get('step') == 'select_style':
#         if message in ['기본형', '이미지형', '아이템리스트형']:
#             state['selected_style'] = message
#         else: # 기본 스타일을 fallback으로 사용
#             state['selected_style'] = '기본형'
        
#         # 템플릿 생성 및 검증 로직 실행
#         state['step'] = 'generate_and_validate'
#         return process_chat_message(message, state)

#     if state.get('step') == 'generate_and_validate':
#         template_draft = generate_template(state['original_request'], state.get('selected_style', '기본형'))
#         state['template_draft'] = template_draft
#         validation_result = validate_template(template_draft)
#         state['validation_result'] = validation_result
#         state['correction_attempts'] = 0

#         if validation_result['status'] == 'accepted':
#             state['step'] = 'completed'
#             return process_chat_message(message, state) # 완료 단계로 이동
#         else:
#             state['step'] = 'correction'
#             return {
#                 'message': f'템플릿을 생성했지만 규정 위반이 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result["suggestion"]}\n\nAI가 자동으로 수정하겠습니다.',
#                 'state': state
#             }
            
#     elif state['step'] == 'correction':
#         if state['correction_attempts'] < MAX_CORRECTION_ATTEMPTS:
#             corrected_template = correct_template(state)
#             state['template_draft'] = corrected_template
#             state['correction_attempts'] += 1
#             validation_result = validate_template(corrected_template)
#             state["validation_result"] = validation_result
            
#             if validation_result["status"] == "accepted":
#                 state["step"] = "completed"
#                 return process_chat_message(message, state) # 완료 단계로 이동
#             else:
#                 return process_chat_message(message, state) # 재귀 호출로 자동 수정 반복
#         else:
#             state['step'] = 'manual_correction'
#             return {
#                 'message': f'AI 자동 수정이 {MAX_CORRECTION_ATTEMPTS}회 모두 실패했습니다.\n\n현재 템플릿:\n{state["template_draft"]}\n\n마지막 문제점: {state["validation_result"]["reason"]}\n\n직접 수정하시겠습니까? 수정할 내용을 입력해주세요.',
#                 'state': state,
#                 'options': ['포기하기']
#             }
            
#     elif state['step'] == 'manual_correction':
#         if message == '포기하기':
#             state['step'] = 'initial'
#             return {'message': '템플릿 생성을 포기했습니다. 새로운 요청을 입력해주세요.', 'state': {'step': 'initial'}}
#         else:
#             state['template_draft'] = message
#             validation_result = validate_template(message)
#             state['validation_result'] = validation_result
#             if validation_result['status'] == 'accepted':
#                 state['step'] = 'completed'
#                 return process_chat_message(message, state) # 완료 단계로 이동
#             else:
#                 return {
#                     'message': f'🚨 수정하신 템플릿에도 여전히 문제가 있습니다.\n\n문제점: {validation_result["reason"]}\n\n다시 수정해주시거나 "포기하기"를 선택해주세요.',
#                     'state': state,
#                     'options': ['포기하기']
#                 }
    
#     elif state['step'] == 'completed':
#         final_template = state.get("template_draft", "")
#         html_preview = render_final_template(final_template)
#         parameterized_result = parameterize_template(final_template)
#         return {
#             "message": "AI 수정이 완료되었습니다! 규정을 준수하는 템플릿이 생성되었습니다.",
#             "state": state,
#             "template": final_template,
#             'html_preview': html_preview,
#             'editable_variables': parameterized_result
#         }

#     return { 'message': '알 수 없는 오류가 발생했습니다. 다시 시도해주세요.', 'state': state }


# # ... (generate_template, validate_template, correct_template 함수들은 그대로 유지)
# def generate_template(request: str, style: str) -> str:
#     # 1. 유사한 성공 사례를 검색하는 부분은 기존과 동일합니다.
#     example_docs = retrievers['whitelist'].invoke(request)
#     examples = "\n\n".join([f"예시 {i+1}:\n{doc.page_content}" for i, doc in enumerate(example_docs)])

#     # 2. 모든 스타일 예시와 지시사항이 포함된 단일 프롬프트로 수정합니다.
#     expansion_prompt = ChatPromptTemplate.from_template(
#         """당신은 사용자의 핵심 의도와 '선택된 스타일'을 바탕으로, 정보가 풍부한 알림톡 템플릿 초안을 확장하는 전문가입니다.
#         당신의 유일한 임무는 아래 지시사항에 따라 **정보가 확장된 템플릿 초안 하나만**을 생성하는 것입니다. 초안 외에 다른 설명은 절대로 덧붙이지 마세요.

#         # 지시사항
#         1. '사용자 핵심 의도'와 '선택된 스타일'({style})을 확인하세요.
#         2. 아래 '스타일 유형별 예시'에서 '선택된 스타일'과 일치하는 예시를 찾아 그 구조와 톤앤매너를 학습하세요.
#         3. 학습한 스타일 구조에 '사용자 핵심 의도'와 '유사한 성공 사례'의 내용을 조합하여 완전한 템플릿 초안을 만드세요.
#         4. '유사한 성공 사례'를 참고하여, 어떤 정보(예: 지원 대상, 신청 기간 등)를 추가해야 할지 **추론**하고, 적절한 #{{변수}}를 사용하세요.

#         # 사용자 핵심 의도: {original_request}
#         # 선택된 스타일: {style}
#         # 유사한 성공 사례 (참고용): {examples}

#         # --- 스타일 유형별 예시 (참고용) ---

#         ## 1. 기본형
#         - 특징: 간결한 제목과 핵심 내용을 담은 한두 문장으로 구성되며, 변수를 사용하여 정보를 명확하게 전달합니다.
#         - 예시 (급여명세서 알림):
#         #{제목}
#         #{발신 스페이스}에서 발송한 #{수신자}님의 급여명세서가 도착했습니다.
#         #{추가 메시지}
#         급여명세서 확인

#         ## 2. 이미지형
#         - 특징: 템플릿에 내용과 관련된 아이콘이나 이미지가 포함되는 것을 가정하며, 정중하고 친절한 어조의 안내 문장으로 구성됩니다.
#         - 예시 (수강료 납부 현황 안내):
#         수강료 납부 현황 안내
#         안녕하세요, #{발신 스페이스}입니다.
#         #{수신자명}님의 #{교육비납부월}월 수강료 납부가 확인되었습니다.
#         항상 #{발신 스페이스}을(를) 이용해 주셔서 감사드리며, 앞으로도 최선을 다하겠습니다.
#         감사합니다.

#         ## 3. 아이템리스트형
#         - 특징: 인사말과 안내 문장 아래에 주요 정보를 목록(리스트) 형태로 나열하며, 각 항목은 '이름: 값'의 형태를 가집니다.
#         - 예시 (행사 안내):
#         행사 안내
#         안녕하세요 #{수신자명} 고객님, #{행사명} 행사에 참여해주셔서 감사합니다.
#         ▶ 성함 : #{성함}
#         ▶ 날짜 : #{날짜}
#         ▶ 장소 : #{장소}
#         날짜와 장소를 확인해주세요.
#         ※ 해당 메시지는 고객님의 알림 신청에 의해 발송되는 일회성 메시지입니다.

#         # ------------------------------------
        
#         # 확장된 템플릿 초안 (오직 템플릿 텍스트만 출력):"""
#     )

#     # 3. 체인을 생성하고 실행합니다. (invoke 호출이 다시 단순해졌습니다.)
#     expansion_chain = expansion_prompt | llm | StrOutputParser()
#     expanded_draft = expansion_chain.invoke({
#         "original_request": request,
#         "style": style,
#         "examples": examples
#     })
#     return expanded_draft

# def validate_template(draft: str) -> dict:
#     parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
#     step_back_prompt = ChatPromptTemplate.from_template(
#         """당신은 알림톡 규정 심사관입니다. 주어진 '템플릿 초안'을 보고, 아래 '판단 기준'과 '예시'를 참고하여 이 템플릿에서 규정 위반 소지가 가장 큰 핵심 쟁점 **한 가지**를 명확한 질문 형태로 만들어주세요.

#         # 판단 기준
#         - 광고성 문구(쿠폰, 할인, 특가, 이벤트, 구매 유도 등)가 포함되어 있는가?
#         - 수신 대상이 사전에 동의한 특정 고객으로 명확하게 한정되는가?
#         - 주민등록번호 등 민감한 개인정보를 요구하거나 포함하고 있는가?

#         # 예시
#         - 예시 입력 1: "오늘만 특가! 저희 매장에 방문해서 50% 할인 쿠폰을 받으세요!"
#         - 예시 출력 1: "이 템플릿에 '특가', '할인'과 같은 명백한 광고성 문구가 포함되어 있는가?"

#         - 예시 입력 2: "안녕하세요, #{행사명} 참여 안내입니다."
#         - 예시 출력 2: "이 템플릿의 수신 대상이 사전에 동의한 특정 고객으로 명확히 한정되는가?"

#         - 예시 입력 3: "비밀번호 변경을 위해 주민번호를 입력해주세요."
#         - 예시 출력 3: "이 템플릿이 민감한 개인정보인 주민등록번호를 직접적으로 요구하고 있는가?"

#         # 템플릿 초안
#         {draft}

#         # 핵심 쟁점 (질문 형태로 출력):"""
#     )
    
#     # 1. 체인(Chain)을 먼저 명확하게 정의합니다.
#     step_back_chain = step_back_prompt | llm | StrOutputParser()
#     # 2. 정의된 체인을 실행(invoke)합니다.
#     step_back_question = step_back_chain.invoke({"draft": draft})    
#     compliance_docs = retrievers['compliance'].invoke(step_back_question)
#     rules_with_metadata = "\n\n".join([f"문서 내용: {doc.page_content}" for doc in compliance_docs])
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
#     result = validation_chain.invoke({"draft": draft, "rules": rules_with_metadata, "rejections": rejections, "format_instructions": parser.get_format_instructions()})
#     return result

# def correct_template(state: dict) -> str:
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
#     new_draft = correction_chain.invoke({"original_request": state['original_request'],"rejected_draft": state['template_draft'],"rejection_reason": state['validation_result']['reason'] + "\n개선 제안: " + state['validation_result']['suggestion']})
#     return new_draft

import os
import json
import re
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

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

def render_final_template(template_string: str) -> str:
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
            <div class="body-text">{body}</div>
            <div class="note-text">{note}</div>
            <div class="button-container"><span>{button_text}</span></div>
        </div>
    </div>
    <style>
        .template-preview {{ max-width: 350px; border-radius: 8px; overflow: hidden; font-family: 'Malgun Gothic', '맑은 고딕', sans-serif; border: 1px solid #e0e0e0; margin: 1em 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .template-preview .header {{ background-color: #F0CA4F; color: #333; padding: 10px 15px; font-weight: bold; font-size: 14px; }}
        .template-preview .content {{ background-color: #E6ECF2; padding: 25px 20px; position: relative; }}
        .template-preview .icon {{ position: absolute; top: 25px; right: 20px; font-size: 36px; opacity: 0.5; }}
        .template-preview .title {{ font-size: 24px; font-weight: bold; margin: 0 0 20px; padding-right: 40px; color: #333; }}
        .template-preview .body-text {{ font-size: 15px; line-height: 1.6; color: #555; margin-bottom: 20px; }}
        .template-preview .note-text {{ font-size: 13px; line-height: 1.5; color: #777; margin-bottom: 25px; }}
        .template-preview .placeholder {{ color: #007bff; font-weight: bold; }}
        .template-preview .button-container {{ background-color: #FFFFFF; border: 1px solid #d0d0d0; border-radius: 5px; text-align: center; padding: 12px 10px; font-size: 15px; font-weight: bold; color: #007bff; cursor: pointer; }}
    </style>
    """
    return html_output

def parameterize_template(template_string: str) -> Dict:
    """템플릿을 매개변수화하는 함수 - 오류 처리 강화"""
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
        # 결과 검증 및 기본값 설정
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
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    data_dir = os.path.join(project_root, 'data')
    approved_templates = load_line_by_line(os.path.join(data_dir, "approved_templates.txt"))
    rejected_templates = load_by_separator(os.path.join(data_dir, "rejected_templates.txt"))
    from chromadb.config import Settings
    docs_compliance = CustomRuleLoader(os.path.join(data_dir, "compliance_rules.txt")).load()
    docs_generation = CustomRuleLoader(os.path.join(data_dir, "generation_rules.txt")).load()
    docs_whitelist = [Document(page_content=t) for t in approved_templates]
    docs_rejected = [Document(page_content=t) for t in rejected_templates]
    """
        추후에 반려된 템플릿 추가하면 고도화 가능
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db_path = os.path.join(project_root, "vector_db")
    client_settings = Settings(anonymized_telemetry=False)
    def create_db(name, docs):
        if docs:
            return Chroma.from_documents(docs, embeddings, collection_name=name, persist_directory=vector_db_path, client_settings=client_settings)
        """
            아래 return Chroma (~~)는 data 에 없는데 사용하고싶은 백터 db 가 있다면 사용.
            불필요한 코드.
        """
        # return Chroma(collection_name=name, embedding_function=embeddings, persist_directory=vector_db_path, client_settings=client_settings)
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
        ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])
        if Ranker:
            compressor = FlashRankRerank(top_n=5)
            return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        return ensemble_retriever
    retrievers['compliance'] = create_hybrid_retriever(db_compliance, docs_compliance)
    retrievers['generation'] = create_hybrid_retriever(db_generation, docs_generation)
    retrievers['whitelist'] = create_hybrid_retriever(db_whitelist, docs_whitelist)
    retrievers['rejected'] = create_hybrid_retriever(db_rejected, docs_rejected)

def process_chat_message(message: str, state: dict) -> dict:
    """채팅 메시지 처리 - 오류 처리 강화"""
    try:
        if state['step'] == 'initial':
            state['original_request'] = message
            state['step'] = 'recommend_templates'
            similar_docs = retrievers['whitelist'].invoke(message)
            
            if not similar_docs:
                state['step'] = 'select_style'
                return {
                    'message': '유사한 기존 템플릿을 찾지 못했습니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:',
                    'state': state,
                    'options': ['기본형', '이미지형', '아이템리스트형']
                }
            
            templates = [f"템플릿 {i+1}:\n{doc.page_content}" for i, doc in enumerate(similar_docs[:3])]
            return {
                'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다:\n\n' + '\n\n'.join(templates) + '\n\n이 중에서 사용하실 템플릿을 선택하시거나, 새로운 템플릿 생성을 원하시면 "신규 생성"을 선택해주세요.',
                'state': state,
                'options': ['템플릿 1', '템플릿 2', '템플릿 3', '신규 생성'],
                'templates': [doc.page_content for doc in similar_docs[:3]]
            }
        
        elif state['step'] == 'recommend_templates':
            if message in ['템플릿 1', '템플릿 2', '템플릿 3']:
                template_idx = int(message.split()[1]) - 1
                similar_docs = retrievers['whitelist'].invoke(state['original_request'])
                state['template_draft'] = similar_docs[template_idx].page_content
            elif message == '신규 생성':
                state['step'] = 'select_style'
                return {
                    'message': '새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요:',
                    'state': state,
                    'options': ['기본형', '이미지형', '아이템리스트형']
                }
            else: # 사용자가 직접 텍스트를 입력한 경우 (스타일 선택으로 바로 넘어감)
                 state['step'] = 'select_style'
                 return process_chat_message(message, state)

        if state.get('step') == 'select_style':
            if message in ['기본형', '이미지형', '아이템리스트형']:
                state['selected_style'] = message
            else: # 기본 스타일을 fallback으로 사용
                state['selected_style'] = '기본형'
            
            # 템플릿 생성 및 검증 로직 실행
            state['step'] = 'generate_and_validate'
            return process_chat_message(message, state)

        if state.get('step') == 'generate_and_validate':
            template_draft = generate_template(state['original_request'], state.get('selected_style', '기본형'))
            state['template_draft'] = template_draft
            validation_result = validate_template(template_draft)
            state['validation_result'] = validation_result
            state['correction_attempts'] = 0

            if validation_result['status'] == 'accepted':
                state['step'] = 'completed'
                return process_chat_message(message, state) # 완료 단계로 이동
            else:
                state['step'] = 'correction'
                return {
                    'message': f'템플릿을 생성했지만 규정 위반이 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result["suggestion"]}\n\nAI가 자동으로 수정하겠습니다.',
                    'state': state
                }
                
        elif state['step'] == 'correction':
            if state['correction_attempts'] < MAX_CORRECTION_ATTEMPTS:
                corrected_template = correct_template(state)
                state['template_draft'] = corrected_template
                state['correction_attempts'] += 1
                validation_result = validate_template(corrected_template)
                state["validation_result"] = validation_result
                
                if validation_result["status"] == "accepted":
                    state["step"] = "completed"
                    return process_chat_message(message, state) # 완료 단계로 이동
                else:
                    return process_chat_message(message, state) # 재귀 호출로 자동 수정 반복
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
                state['template_draft'] = message
                validation_result = validate_template(message)
                state['validation_result'] = validation_result
                if validation_result['status'] == 'accepted':
                    state['step'] = 'completed'
                    return process_chat_message(message, state) # 완료 단계로 이동
                else:
                    return {
                        'message': f'🚨 수정하신 템플릿에도 여전히 문제가 있습니다.\n\n문제점: {validation_result["reason"]}\n\n다시 수정해주시거나 "포기하기"를 선택해주세요.',
                        'state': state,
                        'options': ['포기하기']
                    }
        
        elif state['step'] == 'completed':
            final_template = state.get("template_draft", "")
            html_preview = render_final_template(final_template)
            parameterized_result = parameterize_template(final_template)
            
            # 안전한 변수 접근
            parameterized_template = parameterized_result.get("parameterized_template", final_template)
            variables = parameterized_result.get("variables", [])
            
            # editable_variables 구성
            editable_variables = {
                "parameterized_template": parameterized_template,
                "variables": variables
            } if variables else None
            
            return {
                'message': '✅ 템플릿이 성공적으로 생성되었습니다!',
                'state': state,
                'template': final_template,
                'html_preview': html_preview,
                'editable_variables': editable_variables
            }
        
        # 기본 fallback
        return {
            'message': '알 수 없는 상태입니다. 다시 시도해주세요.',
            'state': {'step': 'initial'}
        }
        
    except Exception as e:
        print(f"Error in process_chat_message: {e}")
        return {
            'message': f'처리 중 오류가 발생했습니다: {str(e)}',
            'state': {'step': 'initial'}
        }

def generate_template(request: str, style: str = "기본형") -> str:
    """템플릿 생성 함수"""
    try:
        generation_docs = retrievers['generation'].invoke(request)
        generation_rules = "\n".join([doc.page_content for doc in generation_docs])
        
        generation_prompt = ChatPromptTemplate.from_template(
            """당신은 알림톡 템플릿 생성 전문가입니다. 사용자의 요청에 따라 적절한 알림톡 템플릿을 생성해주세요.
            # 사용자 요청: {request}
            # 선택된 스타일: {style}
            # 생성 규칙:
            {rules}
            # 지시사항:
            1. 위 규칙을 준수하여 알림톡 템플릿을 생성하세요.
            2. 템플릿은 제목, 본문, 버튼 텍스트로 구성되어야 합니다.
            3. 각 줄은 개행으로 구분하고, 마지막 줄은 버튼 텍스트입니다.
            4. 템플릿 텍스트만 출력하고, 다른 설명은 추가하지 마세요.
            """
        )
        
        generation_chain = generation_prompt | llm | StrOutputParser()
        template = generation_chain.invoke({
            "request": request,
            "style": style,
            "rules": generation_rules
        })
        
        return template.strip()
    except Exception as e:
        print(f"Error in generate_template: {e}")
        return "템플릿 생성 중 오류가 발생했습니다.\n다시 시도해주세요.\n확인"

def validate_template(draft: str) -> dict:
    """템플릿 검증 함수"""
    try:
        parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
        compliance_docs = retrievers['compliance'].invoke(draft)
        rejected_docs = retrievers['rejected'].invoke(draft)
        
        rules_with_metadata = "\n".join([f"[규칙 ID: {doc.metadata.get('rule_id', 'unknown')}] {doc.page_content}" for doc in compliance_docs])
        rejections = "\n".join([doc.page_content for doc in rejected_docs])
        
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
    except Exception as e:
        print(f"Error in validate_template: {e}")
        return {
            "status": "accepted",
            "reason": "검증 중 오류가 발생했지만 템플릿을 승인합니다.",
            "evidence": None,
            "suggestion": None,
            "revised_template": draft
        }

def correct_template(state: dict) -> str:
    """템플릿 수정 함수"""
    try:
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
        return new_draft.strip()
    except Exception as e:
        print(f"Error in correct_template: {e}")
        return state.get('template_draft', '수정 중 오류가 발생했습니다.')

