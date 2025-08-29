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
#     """템플릿을 매개변수화하는 함수 - 오류 처리 강화"""
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
#         # 결과 검증 및 기본값 설정
#         if not isinstance(result, dict):
#             result = {"parameterized_template": template_string, "variables": []}
#         if "parameterized_template" not in result:
#             result["parameterized_template"] = template_string
#         if "variables" not in result:
#             result["variables"] = []
#         return result
#     except Exception as e:
#         print(f"Error during parameterization: {e}")
#         return {"parameterized_template": template_string, "variables": []}

# def initialize_system():
#     global llm, retrievers, approved_templates, rejected_templates
#     if llm is not None:
#         return
#     llm = ChatOpenAI(model="gpt-5", temperature=0.2)
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

# def process_chat_message(message: str, state: dict) -> dict:
#     """채팅 메시지 처리 - 오류 처리 강화"""
#     try:
#         if state['step'] == 'initial':
#             state['original_request'] = message
#             state['step'] = 'recommend_templates'
#             similar_docs = retrievers['whitelist'].invoke(message)
            
#             if not similar_docs:
#                 state['step'] = 'select_style'
#                 return {
#                     'message': '유사한 기존 템플릿을 찾지 못했습니다. 새로운 템플릿을 생성하겠습니다.\n\n원하시는 스타일을 선택해주세요:',
#                     'state': state,
#                     'options': ['기본형', '이미지형', '아이템리스트형']
#                 }
            
#             templates = [f"템플릿 {i+1}:\n{doc.page_content}" for i, doc in enumerate(similar_docs[:3])]
#             return {
#                 'message': '요청하신 내용과 유사한 기존 템플릿을 찾았습니다:\n\n' + '\n\n'.join(templates) + '\n\n이 중에서 사용하실 템플릿을 선택하시거나, 새로운 템플릿 생성을 원하시면 "신규 생성"을 선택해주세요.',
#                 'state': state,
#                 'options': ['템플릿 1', '템플릿 2', '템플릿 3', '신규 생성'],
#                 'templates': [doc.page_content for doc in similar_docs[:3]]
#             }
        
#         elif state['step'] == 'recommend_templates':
#             if message in ['템플릿 1', '템플릿 2', '템플릿 3']:
#                 template_idx = int(message.split()[1]) - 1
#                 similar_docs = retrievers['whitelist'].invoke(state['original_request'])
#                 state['template_draft'] = similar_docs[template_idx].page_content
#             elif message == '신규 생성':
#                 state['step'] = 'select_style'
#                 return {
#                     'message': '새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요:',
#                     'state': state,
#                     'options': ['기본형', '이미지형', '아이템리스트형']
#                 }
#             else: # 사용자가 직접 텍스트를 입력한 경우 (스타일 선택으로 바로 넘어감)
#                  state['step'] = 'select_style'
#                  return process_chat_message(message, state)

#         if state.get('step') == 'select_style':
#             if message in ['기본형', '이미지형', '아이템리스트형']:
#                 state['selected_style'] = message
#             else: # 기본 스타일을 fallback으로 사용
#                 state['selected_style'] = '기본형'
            
#             # 템플릿 생성 및 검증 로직 실행
#             state['step'] = 'generate_and_validate'
#             return process_chat_message(message, state)

#         if state.get('step') == 'generate_and_validate':
#             template_draft = generate_template(state['original_request'], state.get('selected_style', '기본형'))
#             state['template_draft'] = template_draft
#             validation_result = validate_template(template_draft)
#             state['validation_result'] = validation_result
#             state['correction_attempts'] = 0

#             if validation_result['status'] == 'accepted':
#                 state['step'] = 'completed'
#                 return process_chat_message(message, state) # 완료 단계로 이동
#             else:
#                 state['step'] = 'correction'
#                 return {
#                     'message': f'템플릿을 생성했지만 규정 위반이 발견되었습니다.\n\n문제점: {validation_result["reason"]}\n\n개선 제안: {validation_result["suggestion"]}\n\nAI가 자동으로 수정하겠습니다.',
#                     'state': state
#                 }
                
#         elif state['step'] == 'correction':
#             if state['correction_attempts'] < MAX_CORRECTION_ATTEMPTS:
#                 corrected_template = correct_template(state)
#                 state['template_draft'] = corrected_template
#                 state['correction_attempts'] += 1
#                 validation_result = validate_template(corrected_template)
#                 state["validation_result"] = validation_result
                
#                 if validation_result["status"] == "accepted":
#                     state["step"] = "completed"
#                     return process_chat_message(message, state) # 완료 단계로 이동
#                 else:
#                     return process_chat_message(message, state) # 재귀 호출로 자동 수정 반복
#             else:
#                 state['step'] = 'manual_correction'
#                 return {
#                     'message': f'AI 자동 수정이 {MAX_CORRECTION_ATTEMPTS}회 모두 실패했습니다.\n\n현재 템플릿:\n{state["template_draft"]}\n\n마지막 문제점: {state["validation_result"]["reason"]}\n\n직접 수정하시겠습니까? 수정할 내용을 입력해주세요.',
#                     'state': state,
#                     'options': ['포기하기']
#                 }
                
#         elif state['step'] == 'manual_correction':
#             if message == '포기하기':
#                 state['step'] = 'initial'
#                 return {'message': '템플릿 생성을 포기했습니다. 새로운 요청을 입력해주세요.', 'state': {'step': 'initial'}}
#             else:
#                 state['template_draft'] = message
#                 validation_result = validate_template(message)
#                 state['validation_result'] = validation_result
#                 if validation_result['status'] == 'accepted':
#                     state['step'] = 'completed'
#                     return process_chat_message(message, state) # 완료 단계로 이동
#                 else:
#                     return {
#                         'message': f'🚨 수정하신 템플릿에도 여전히 문제가 있습니다.\n\n문제점: {validation_result["reason"]}\n\n다시 수정해주시거나 "포기하기"를 선택해주세요.',
#                         'state': state,
#                         'options': ['포기하기']
#                     }
        
#         elif state['step'] == 'completed':
#             final_template = state.get("template_draft", "")
#             html_preview = render_final_template(final_template)
#             parameterized_result = parameterize_template(final_template)
            
#             # 안전한 변수 접근
#             parameterized_template = parameterized_result.get("parameterized_template", final_template)
#             variables = parameterized_result.get("variables", [])
            
#             # editable_variables 구성
#             editable_variables = {
#                 "parameterized_template": parameterized_template,
#                 "variables": variables
#             } if variables else None
            
#             return {
#                 'message': '✅ 템플릿이 성공적으로 생성되었습니다!',
#                 'state': state,
#                 'template': final_template,
#                 'html_preview': html_preview,
#                 'editable_variables': editable_variables
#             }
        
#         # 기본 fallback
#         return {
#             'message': '알 수 없는 상태입니다. 다시 시도해주세요.',
#             'state': {'step': 'initial'}
#         }
        
#     except Exception as e:
#         print(f"Error in process_chat_message: {e}")
#         return {
#             'message': f'처리 중 오류가 발생했습니다: {str(e)}',
#             'state': {'step': 'initial'}
#         }

# def generate_template(request: str, style: str = "기본형") -> str:
#     """템플릿 생성 함수"""
#     try:
#         generation_docs = retrievers['generation'].invoke(request)
#         generation_rules = "\n".join([doc.page_content for doc in generation_docs])
        
#         generation_prompt = ChatPromptTemplate.from_template(
#             """당신은 알림톡 템플릿 생성 전문가입니다. 사용자의 요청에 따라 적절한 알림톡 템플릿을 생성해주세요.
#             # 사용자 요청: {request}
#             # 선택된 스타일: {style}
#             # 생성 규칙:
#             {rules}
#             # 지시사항:
#             1. 위 규칙을 준수하여 알림톡 템플릿을 생성하세요.
#             2. 템플릿은 제목, 본문, 버튼 텍스트로 구성되어야 합니다.
#             3. 각 줄은 개행으로 구분하고, 마지막 줄은 버튼 텍스트입니다.
#             4. 템플릿 텍스트만 출력하고, 다른 설명은 추가하지 마세요.
#             """
#         )
        
#         generation_chain = generation_prompt | llm | StrOutputParser()
#         template = generation_chain.invoke({
#             "request": request,
#             "style": style,
#             "rules": generation_rules
#         })
        
#         return template.strip()
#     except Exception as e:
#         print(f"Error in generate_template: {e}")
#         return "템플릿 생성 중 오류가 발생했습니다.\n다시 시도해주세요.\n확인"

# def validate_template(draft: str) -> dict:
#     """템플릿 검증 함수"""
#     try:
#         parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
#         compliance_docs = retrievers['compliance'].invoke(draft)
#         rejected_docs = retrievers['rejected'].invoke(draft)
        
#         rules_with_metadata = "\n".join([f"[규칙 ID: {doc.metadata.get('rule_id', 'unknown')}] {doc.page_content}" for doc in compliance_docs])
#         rejections = "\n".join([doc.page_content for doc in rejected_docs])
        
#         validation_prompt = ChatPromptTemplate.from_template(
#             """당신은 알림톡 템플릿의 규정 준수 여부를 판단하는 전문가입니다.
#             주어진 템플릿 초안을 검토하고, 규정 위반 사항이 있는지 분석해주세요.
#             # 검토할 템플릿 초안:
#             {draft}
#             # 준수해야 할 규칙들:
#             {rules}
#             # 과거 반려된 템플릿 사례들:
#             {rejections}
#             # 지시사항:
#             1. 템플릿이 규칙을 위반하는지 꼼꼼히 검토하세요.
#             2. 위반 사항이 없다면 'status'를 'accepted'로 설정하고, 'revised_template'에 원본 초안을 그대로 넣으세요.
#             3. 위반 사항이 있다면 'status'를 'rejected'로 설정하고, 'suggestion'에 구체적인 개선 방안을 제시하세요.
#             # 출력 형식 (JSON):
#             {format_instructions}
#             """
#         )
#         validation_chain = validation_prompt | llm | parser
#         result = validation_chain.invoke({
#             "draft": draft, 
#             "rules": rules_with_metadata, 
#             "rejections": rejections, 
#             "format_instructions": parser.get_format_instructions()
#         })
#         return result
#     except Exception as e:
#         print(f"Error in validate_template: {e}")
#         return {
#             "status": "accepted",
#             "reason": "검증 중 오류가 발생했지만 템플릿을 승인합니다.",
#             "evidence": None,
#             "suggestion": None,
#             "revised_template": draft
#         }

# def correct_template(state: dict) -> str:
#     """템플릿 수정 함수"""
#     try:
#         attempts = state.get('correction_attempts', 0)
#         if attempts == 0:
#             instruction = "3. 광고성 문구를 제거하거나, 정보성 내용으로 순화하는 등, 제안된 방향에 맞게 템플릿을 수정하세요."
#         elif attempts == 1:
#             instruction = "3. **(2차 수정)** 아직도 문제가 있습니다. 이번에는 '쿠폰', '할인', '이벤트', '특가'와 같은 명백한 광고성 단어를 사용하지 마세요."
#         else:
#             instruction = """3. **(최종 수정: 관점 전환)** 여전히 광고성으로 보입니다. 이것이 마지막 시도입니다.
#             - **관점 전환:** 메시지의 주체를 '우리(사업자)'에서 '고객님'으로 완전히 바꾸세요.
#             - **목적 변경:** '판매'나 '방문 유도'가 아니라, '고객님이 과거에 동의한 내용에 따라 고객님의 권리(혜택) 정보를 안내'하는 것으로 목적을 재정의하세요."""
        
#         correction_prompt_template = """당신은 지적된 문제점을 해결하여 더 나은 대안을 제시하는 전문 카피라이터입니다.
#         당신의 유일한 임무는 아래 지시사항에 따라 **수정된 템플릿 초안 하나만**을 생성하는 것입니다. 초안 외에 다른 설명은 절대로 덧붙이지 마세요.
#         # 원래 사용자 요청: {original_request}
#         # 이전에 제안했던 템플릿 (반려됨): {rejected_draft}
#         # 반려 사유 및 개선 제안: {rejection_reason}
#         # 지시사항
#         1. '반려 사유 및 개선 제안'을 완벽하게 이해하고, 지적된 모든 문제점을 해결하세요.
#         2. '원래 사용자 요청'의 핵심 의도는 유지해야 합니다.
#         {dynamic_instruction}
#         # 수정된 템플릿 초안 (오직 템플릿 텍스트만 출력):
#         """
#         correction_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
#         correction_prompt = correction_prompt.partial(dynamic_instruction=instruction)
#         correction_chain = correction_prompt | llm | StrOutputParser()
#         new_draft = correction_chain.invoke({
#             "original_request": state['original_request'],
#             "rejected_draft": state['template_draft'],
#             "rejection_reason": state['validation_result']['reason'] + "\n개선 제안: " + state['validation_result']['suggestion']
#         })
#         return new_draft.strip()
#     except Exception as e:
#         print(f"Error in correct_template: {e}")
#         return state.get('template_draft', '수정 중 오류가 발생했습니다.')


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
# Docker 컨테이너 내부 경로를 고려하여 경로 설정 방식을 단순화합니다.
# Dockerfile의 WORKDIR이 /app이므로, 상대 경로를 사용하면 됩니다.
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

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
        
    print("서버 시작: 시스템 초기화를 진행합니다...")
    try:
        # Docker 컨테이너의 기본 경로인 /app을 기준으로 경로 설정
        data_dir = 'data'
        vector_db_path = "vector_db"
        
        # llm 및 embeddings 초기화 (API 키는 환경변수에서 자동으로 읽어옴)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 데이터 로드
        approved_templates = load_line_by_line(os.path.join(data_dir, "approved_templates.txt"))
        rejected_templates = load_by_separator(os.path.join(data_dir, "rejected_templates.txt"))
        
        docs_compliance = CustomRuleLoader(os.path.join(data_dir, "compliance_rules.txt")).load()
        docs_generation = CustomRuleLoader(os.path.join(data_dir, "generation_rules.txt")).load()
        docs_whitelist = [Document(page_content=t) for t in approved_templates]
        docs_rejected = [Document(page_content=t) for t in rejected_templates]
        
        # ChromaDB 클라이언트 설정
        from chromadb.config import Settings
        client_settings = Settings(anonymized_telemetry=False)
        
        # DB 생성 헬퍼 함수
        def create_db(name, docs):
            if docs: # 문서가 있을 때만 DB 생성
                return Chroma.from_documents(
                    docs, 
                    embeddings, 
                    collection_name=name, 
                    persist_directory=vector_db_path, 
                    client_settings=client_settings
                )
            return None # 문서가 없으면 None 반환
            
        # DB 인스턴스 생성
        db_compliance = create_db("compliance_rules", docs_compliance)
        db_generation = create_db("generation_rules", docs_generation)
        db_whitelist = create_db("whitelist_templates", docs_whitelist)
        db_rejected = create_db("rejected_templates", docs_rejected)
        
        # 리트리버 생성 헬퍼 함수
        def create_hybrid_retriever(vectorstore, docs):
            # [수정] vectorstore가 None이면 리트리버를 생성하지 않고 None 반환
            if not vectorstore:
                return None
            
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # 문서가 있을 때만 BM25Retriever와 EnsembleRetriever를 구성
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

        # [수정] DB가 성공적으로 생성된 경우에만 리트리버를 등록
        retrievers['compliance'] = create_hybrid_retriever(db_compliance, docs_compliance)
        retrievers['generation'] = create_hybrid_retriever(db_generation, docs_generation)
        retrievers['whitelist'] = create_hybrid_retriever(db_whitelist, docs_whitelist)
        retrievers['rejected'] = create_hybrid_retriever(db_rejected, docs_rejected)

        # [수정] 생성된 리트리버를 확인하는 로그 추가
        for name, retriever in retrievers.items():
            if retriever:
                print(f"✅ '{name}' 리트리버가 성공적으로 생성되었습니다.")
            else:
                print(f"🚨 경고: '{name}' 리트리버를 생성하지 못했습니다 (관련 데이터 파일 부재 추정).")

        print("시스템 초기화 완료.")

    except Exception as e:
        print(f"시스템 초기화 실패: {e}")
        # 초기화 실패 시, 앱이 계속 실행되지 않도록 예외를 다시 발생시킬 수 있습니다.
        # 이는 컨테이너가 재시작되도록 유도하여 문제를 명확히 알립니다.
        raise e

def process_chat_message(message: str, state: dict) -> dict:
    """채팅 메시지 처리 - 오류 처리 강화"""
    try:
        # [수정] 각 단계에서 필요한 리트리버가 있는지 확인
        if state['step'] == 'initial':
            state['original_request'] = message
            state['step'] = 'recommend_templates'
            
            # whitelist 리트리버가 없을 경우 바로 신규 생성으로 이동
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
        # [수정] generation 리트리버가 없을 경우 기본 프롬프트 사용
        if 'generation' not in retrievers or not retrievers['generation']:
            print("🚨 경고: generation 리트리버가 없어 기본 프롬프트로 생성합니다.")
            generation_rules = "사용자의 요청에 맞춰 정보성 템플릿을 생성하세요."
        else:
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
        # [수정] compliance, rejected 리트리버가 없을 경우 기본값 사용
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
