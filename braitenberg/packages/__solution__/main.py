from braitenberg_agent.agent import BraitenbergAgent

from aido_schemas import protocol_agent_DB20, wrap_direct


def main():
    agent = BraitenbergAgent()
    wrap_direct(node=agent, protocol=protocol_agent_DB20)


if __name__ == "__main__":
    main()
