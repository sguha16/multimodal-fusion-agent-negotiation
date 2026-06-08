# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:28:06 2026

@author: sanhi
"""
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
##----------STATE DEFINITIONS-------------##
#State travels through the entire graph, collecting info/changing 
#Sensor Report:
#report from each sensor (radar, cam)-identical structure to report coming fromt he radar agent and camera agent
#Fusion state: this info is needed in the orchestrator and travels through the chain
class SensorReport(TypedDict):
    modality: str
    timestamp: float
    data: Dict[str, Any]
    health: Dict[str, Any]
    confidence: float
    self_assessment: Dict[str, str]
    features: list              # stub fr now
    trend: str
class FusionState(TypedDict):
    radar_report: Optional[SensorReport]
    camera_report: Optional[SensorReport]
    situation_summary: Optional[str]#based on radar and camera report--data
    retrieval_query: Optional[str]#situation_summary reformulated as query for RAG ChromaDB
    retrieved_knowledge: Optional[str]    # RAG output — empty for now--literature
    fusion_strategy: Optional[str]        # mid / late / early-based on data+lit
    decision_explanation: Optional[str]
    iteration_count: Optional[int]
    should_retry_retrieval: Optional[bool]
##-----------NODE DEFINITIONS---------------------##
#1.summarize situation-take reports from radar and cam. Output:situation_summary
#2.retrieve_knowledge:RAG query
#3.decide_fusion:Input:summary+knowledge, output:fusion strategy
#4.explain_decision:input:stategy+situation. output:decision

#---------NODE 1: summarize situation---------------
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3")

def summarize_situation(state: FusionState) -> dict:
    radar = state["radar_report"]
    camera = state["camera_report"]

    prompt = f"""
You are a sensor fusion system monitor.
Given the following sensor reports, write a concise 2-3 sentence summary 
of the current sensor state for a safety engineer.

Radar confidence: {radar['confidence']}, trend: {radar['trend']}
Radar self assessment: {radar['self_assessment']}
Radar health: {radar['health']}

Camera confidence: {camera['confidence']}, trend: {camera['trend']}
Camera self assessment: {camera['self_assessment']}
Camera health: {camera['health']}

Be specific about confidence levels, trends, and any issues.
"""
    situation_summary = llm.invoke(prompt).strip()
    #formulate a query from the situation summary
    query_prompt = f"""
Given this sensor situation:
{situation_summary}

Formulate a concise search query (max 15 words) to retrieve relevant research 
on the best fusion strategy for this situation.
Respond with just the query, nothing else.
"""
    retrieval_query = llm.invoke(query_prompt).strip()
    return {
        "situation_summary": situation_summary,
        "retrieval_query": retrieval_query
    }
#---------NODE 1 END

#---------NODE 2: retreive Knowledge---------------------
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma

# embeddings = OllamaEmbeddings(model="nomic-embed-text")
# vectorstore = Chroma(
#     persist_directory="./fusion_knowledge_base",
#     embedding_function=embeddings
# )

# def retrieve_knowledge(state: FusionState) -> dict:
#     query = state["retrieval_query"]
#     results = vectorstore.similarity_search(query, k=4)
#     retrieved = "\n\n".join([doc.page_content for doc in results])#4 most similar chucls from current knowledge DB joined together as 1 string is returned
#     return {"retrieved_knowledge": retrieved}
    #-----------------------------------------------
    # stub — will be replaced with real ChromaDB RAG
    #return {"retrieved_knowledge": "stub: no knowledge retrieved yet"}
#GRAPH-BASED
from fusion_knowledge_graph import build_fusion_graph
from graph_query import query_fusion_graph

fusion_graph_kg = build_fusion_graph()

def retrieve_knowledge(state: FusionState) -> dict:
    reasoning = query_fusion_graph(
        fusion_graph_kg,
        state["radar_report"],
        state["camera_report"]
    )
    return {"retrieved_knowledge": reasoning}

#---------NODE 2 END

#---------NODE 3: decide fusion and explain---------------------
def decide_fusion(state: FusionState) -> dict:
    situation = state["situation_summary"]
    knowledge = state["retrieved_knowledge"]
    radar_conf = state["radar_report"]["confidence"]
    camera_conf = state["camera_report"]["confidence"]

    prompt = f"""
You are an ADAS sensor fusion decision system.
Current sensor situation:
{situation}
Relevant knowledge from research:
{knowledge}
Radar confidence: {radar_conf}
Camera confidence: {camera_conf}
Based on the sensor analysis above, decide which data sources to use for fusion and why.
Respond in two parts:
Line 1: list the data sources to use, comma separated (from: radar_cube, point_cloud, radar_object_list, rgb_image, camera_bboxes)
Lines 2-6: explain the decision in up to 5 lines. State which sources you are using, which you are ignoring and why, and how the chosen sources complement each other.
"""
##old prompt-- Based on the sensor situation and the research knowledge, decide which fusion strategy to use.
# Choose exactly one of: early, mid, late.
# Respond in two parts:
# Line 1: just one word: early, mid, or late.
# Lines 2-6: explain the decision in up to 5 lines. Reference the research knowledge, 
# describe the current sensor situation, and explain why this fusion strategy was chosen.
    lines = llm.invoke(prompt).strip().lower().split("\n")
    lines = [l.strip() for l in lines if l.strip()]
    sources = lines[0]
    explanation = "\n".join(lines[1:])

    return {
        "fusion_strategy": sources,
        "decision_explanation": explanation
    }
    # strategy = lines[0]
    # explanation = "\n".join(lines[1:])

    # if strategy not in ["early", "mid", "late"]:
    #     strategy = "mid"

    # return {
    #     "fusion_strategy": strategy,
    #     "decision_explanation": explanation
    # }
#---------NODE 3 END-----------------


###BUILD GRAPH
from langgraph.graph import StateGraph, END

graph = StateGraph(FusionState)

graph.add_node("summarize_situation", summarize_situation)
graph.add_node("retrieve_knowledge", retrieve_knowledge)
graph.add_node("decide_fusion", decide_fusion)

graph.add_edge("summarize_situation", "retrieve_knowledge")
graph.add_edge("retrieve_knowledge", "decide_fusion")
graph.add_edge("decide_fusion", END)

graph.set_entry_point("summarize_situation")

app = graph.compile()



# =========================================================================
# TEST: run when executed directly
# =========================================================================
if __name__ == "__main__":
    from radial_loader import RadialLoader
    from radar_agent import RadarAgent
    from camera_agent import CameraAgent

    SEQ = r"C:\Users\sanhi\Downloads\RECORD@2020-11-21_11.54.31"
    CALIB = r"C:\Users\sanhi\RADIal_code\SignalProcessing\CalibrationTable.npy"
    LABELS = r"C:\Users\sanhi\Downloads\labels_CVPR.csv"
    RADIAL_CODE = r"C:\Users\sanhi\RADIal_code"

    loader = RadialLoader(SEQ, CALIB, LABELS, radial_code_path=RADIAL_CODE)
    frame = loader.get_frame(index=0)

    radar_agent = RadarAgent(memory_size=10)
    camera_agent = CameraAgent(memory_size=10)

    radar = radar_agent.get_report(frame["radar"])
    camera = camera_agent.get_report(frame["camera"])

    initial_state = {
        "radar_report": radar,
        "camera_report": camera,
        "situation_summary": None,
        "retrieved_knowledge": None,
        "fusion_strategy": None,
        "decision_explanation": None,
        "iteration_count": 0,
        "should_retry_retrieval": False
    }

    result = app.invoke(initial_state)

    print("\n[SITUATION SUMMARY]")
    print(result["situation_summary"])
    print("\n[RETRIEVED KNOWLEDGE]")
    print(result["retrieved_knowledge"])
    print("\n[FUSION STRATEGY]")
    print(result["fusion_strategy"])
    print("\n[DECISION EXPLANATION]")
    print(result["decision_explanation"])