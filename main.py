# This will hold our graph implementation
# We will use the LangGraph library to create a graph and define the nodes and edges
from typing import List, Sequence
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
# END is a constant that holds the _end_, which is the key for langGraph default ending note,
# when we reach the node with this key the langGraph stops the execution
# MessageGraph is a type of graph that its state is simply a sequence of messages
# And in our graph, every node will receive as an input a list of messages
from langgraph.graph import END, MessageGraph
# these chains are going to run in each node in our langGraph Graph
from chains import generate_chain, reflect_chain

# these two constants are simply going to be the keys of our langGraph nodes we're going to create
REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    # The state is just a list of messages.
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("Hello LangGraph!")

    inputs = HumanMessage(content="""Make this tweet better:
    🔪 It's really EASY TO KILL new ideas in the beginning❗️❗️

Every day I see how creative people bring new, fresh ideas, which are very easy to object to and discard.

But when a good new idea comes along, that's just the core of the idea, a thread to pull on, a thread that doesn't 
need to be cut, but worked on together to bring it to reality.

Jeff Bezos and Lex Fridman discuss this more deeply in the podcast I would like to share with you now.

All the information that Jeff has shared with us is priceless 🏄‍♂️""")

    graph.invoke(inputs)
