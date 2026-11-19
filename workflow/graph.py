from state import State
from supervisor.supervisor import supervisor
from agents.research_agent import research_agent
from agents.data_analysis_agent import data_agent
from agents.coding_agent import coding_agent
from agents.writing_agent import writing_agent
from workflow.hitl import human_input_node
from workflow.route_task import route_task
from workflow.task_router import task_router
from langgraph.graph import StateGraph,START,END


graph = StateGraph(State)

graph.add_node('supervisor',supervisor)
graph.add_node('task_router',task_router)
graph.add_node('research_agent',research_agent)
graph.add_node('data_agent',data_agent)
graph.add_node('coding_agent',coding_agent)
graph.add_node('writing_agent',writing_agent)
graph.add_node('human_input',human_input_node)

graph.add_edge(START,'supervisor')
graph.add_edge('supervisor','task_router')
graph.add_conditional_edges('task_router',route_task,{
    'human_input':'human_input',
    'finish':END
})
graph.add_edge("human_input", "task_router")
graph.add_edge('research_agent','task_router')
graph.add_edge('data_agent','task_router')
graph.add_edge('coding_agent','task_router')
graph.add_edge('writing_agent','task_router')
