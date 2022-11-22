from collision_checker import CollisionChecker

from aido_schemas import protocol_collision_checking, wrap_direct


def main():
    agent = CollisionChecker()
    wrap_direct(node=agent, protocol=protocol_collision_checking)


if __name__ == "__main__":
    main()
