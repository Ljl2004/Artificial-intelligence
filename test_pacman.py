import layout

map_name = "mediumClassic"

pacman_layout = layout.getLayout(map_name)

print(f"地图 '{map_name}' 中的智能体初始位置如下：\n")

for agent in pacman_layout.agentPositions:
    agent_type, pos = agent
    x, y = pos

    if agent_type == 0:
        print(f"吃豆人的初始位置是：(x={x}, y={y})")
    else:
        print(f"幽灵 #{agent_type} 的初始位置是：(x={x}, y={y})")