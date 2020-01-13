import argparse
import json
from generate_json_from_grid import gridToRoadnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rowNum", type=int, default=1)
    parser.add_argument("--colNum", type=int, default=1)
    parser.add_argument("--rowDistance", type=int, default=300)
    parser.add_argument("--columnDistance", type=int, default=300)
    parser.add_argument("--numLeftLanes", type=int, default=1)
    parser.add_argument("--numStraightLanes", type=int, default=2)
    parser.add_argument("--numRightLanes", type=int, default=0)
    parser.add_argument("--laneMaxSpeed", type=float, default=16.67)
    parser.add_argument("--vehLen", type=float, default=5.0)
    parser.add_argument("--vehWidth", type=float, default=2.0)
    parser.add_argument("--vehMaxPosAcc", type=float, default=2.0)
    parser.add_argument("--vehMaxNegAcc", type=float, default=4.5)
    parser.add_argument("--vehUsualPosAcc", type=float, default=2.0)
    parser.add_argument("--vehUsualNegAcc", type=float, default=4.5)
    parser.add_argument("--vehMinGap", type=float, default=2.5)
    parser.add_argument("--vehMaxSpeed", type=float, default=16.67)
    parser.add_argument("--vehHeadwayTime", type=float, default=1.5)
    parser.add_argument("--dir", type=str, default="../data/tmp/")
    parser.add_argument("--roadnetFile", type=str)
    parser.add_argument("--turn", action="store_true")
    parser.add_argument("--tlPlan", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--flowFile", type=str)

    parser.add_argument("--interval_ratio", type=float, default=1.0)
    return parser.parse_args()


def generate_route(row_num, col_num, turn=False):
    ##TODO grid_num
    routes = []
    move = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    # go east x+1 y+0
    # go north x+0 y+1
    # go west x-1 y+0
    # go south x+0 y-1

    def get_straight_route_row(start, direction):
        x, y = start
        route = []
        for _ in range(col_num + 1):
            route.append("road_%d_%d_%d" % (x, y, direction))
            x += move[direction][0]
            y += move[direction][1]
        return route

    def get_straight_route_col(start, direction):
        x, y = start
        route = []
        for _ in range(row_num + 1):
            route.append("road_%d_%d_%d" % (x, y, direction))
            x += move[direction][0]
            y += move[direction][1]
        return route

    for i in range(1, row_num + 1):
        routes.append(get_straight_route_row((0, i), 0))
        routes.append(get_straight_route_row((col_num + 1, i), 2))
    for i in range(1, col_num + 1):
        routes.append(get_straight_route_col((i, 0), 1))
        routes.append(get_straight_route_col((i, row_num + 1), 3))

    if turn:
        grid_num = 1 # tmp for single inter_section
        def get_turn_route(start, direction):
            x, y = start
            route = []
            cur = 0
            for _ in range(grid_num * 2): #tmep gridNum
                route.append("road_%d_%d_%d" % (x, y, direction[cur]))
                x += move[direction[cur]][0]
                y += move[direction[cur]][1]
                cur = 1 - cur
            return route
        #routes.append(get_turn_route((1, 0), (1, 0))) # right
        routes.append(get_turn_route((0, 1), (0, 1))) # left
        routes.append(get_turn_route((grid_num + 1, grid_num), (2, 3))) # left
        routes.append(get_turn_route((1, grid_num + 1), (3, 0)))  # left
        routes.append(get_turn_route((grid_num, 0), (1, 2)))  # left
        #routes.append(get_turn_route((grid_num, grid_num + 1), (3, 2))) # right
        #routes.append(get_turn_route((0, grid_num), (0, 3))) # right

        #routes.append(get_turn_route((grid_num + 1, 1), (2, 1))) #right


    return routes


if __name__ == '__main__':
    args = parse_args()
    if args.roadnetFile is None:
        args.roadnetFile = "roadnet_%d_%d%s.json" % (args.rowNum, args.colNum, "")
    if args.flowFile is None:
        args.flowFile = "flow_%d_%d%s.json" % (args.rowNum, args.colNum, "")

    args.flowFiles = []

    for i in range(100, 1100, 100):
        args.flowFiles.append(str("flow_%d_%d_%d%s.json" % (args.rowNum, args.colNum, i, "")))

    grid = {
        "rowNumber": args.colNum,
        "columnNumber": args.rowNum,
        "rowDistances": [args.rowDistance] * (args.colNum - 1),
        "columnDistances": [args.columnDistance] * (args.rowNum - 1),
        "outRowDistance": args.rowDistance,
        "outColumnDistance": args.columnDistance,
        "intersectionWidths": [[(2 + 3 * (args.numLeftLanes + args.numStraightLanes + args.numRightLanes))] * args.rowNum] * args.colNum,
        "numLeftLanes": args.numLeftLanes,
        "numStraightLanes": args.numStraightLanes,
        "numRightLanes": args.numRightLanes,
        "laneMaxSpeed": args.laneMaxSpeed,
        "tlPlan": args.tlPlan,
        "laneWidth": 3
    }

    json.dump(gridToRoadnet(**grid), open(args.dir + args.roadnetFile, "w"), indent=2)

    #vehicle_template = {
    #    "length": args.vehLen,
    #    "width": args.vehWidth,
    #    "maxPosAcc": args.vehMaxPosAcc,
    #    "maxNegAcc": args.vehMaxNegAcc,
    #    "usualPosAcc": args.vehUsualPosAcc,
    #    "usualNegAcc": args.vehUsualNegAcc,
    #    "minGap": args.vehMinGap,
    #    "maxSpeed": args.vehMaxSpeed,
    #    "headwayTime": args.vehHeadwayTime
    #}

    #for flow_file in args.flowFiles:
    #    args.interval = float(3600 / int(flow_file.split('_')[3].split('.')[0])) * args.interval_ratio# !!!!!!! multi_phase
    #    routes = generate_route(args.rowNum, args.colNum, args.turn)
    #    flow = []
    #    for route in routes:
    #        flow.append({
    #            "vehicle": vehicle_template,
    #            "route": route,
    #            "interval": args.interval,
    #            "startTime": 0,
    #            "endTime": 3600
    #        })
    #    json.dump(flow, open(args.dir + flow_file, "w"), indent=2)

# python generate_grid_scenario.py --numLeftLanes 0 --numRightLanes 0 --numStraightLanes 1