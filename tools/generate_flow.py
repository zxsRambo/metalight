import argparse
import json
from generate_json_from_grid import gridToRoadnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("gridNum", type=int)
    parser.add_argument("--rowDistance", type=int, default=300)
    parser.add_argument("--columnDistance", type=int, default=300)
    parser.add_argument("--intersectionWidth", type=int, default=10)
    parser.add_argument("--numLeftLanes", type=int, default=1)
    parser.add_argument("--numStraightLanes", type=int, default=1)
    parser.add_argument("--numRightLanes", type=int, default=1)
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
    parser.add_argument("--dir", type=str, default="../data/")
    parser.add_argument("--roadnetFile", type=str)
    parser.add_argument("--turn", action="store_true")
    parser.add_argument("--tlPlan", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--flowFile", type=str)
    return parser.parse_args()


def generate_route(grid_num, turn=False):
    routes = []
    move = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def get_straight_route(start, direction):
        x, y = start
        route = []
        for _ in range(grid_num + 1):
            route.append("road_%d_%d_%d" % (x, y, direction))
            x += move[direction][0]
            y += move[direction][1]
        return route

    for i in range(1, grid_num + 1):
        routes.append(get_straight_route((0, i), 0))
        routes.append(get_straight_route((grid_num + 1, i), 2))
        routes.append(get_straight_route((i, 0), 1))
        routes.append(get_straight_route((i, grid_num + 1), 3))

    if turn:
        def get_turn_route(start, direction):
            x, y = start
            route = []
            cur = 0
            for _ in range(grid_num * 2):
                route.append("road_%d_%d_%d" % (x, y, direction[cur]))
                x += move[direction[cur]][0]
                y += move[direction[cur]][1]
                cur = 1 - cur
            return route

        # turn left
        routes.append(get_turn_route((grid_num + 1, grid_num), (2, 3)))
        routes.append(get_turn_route((0, 1), (0, 1)))
        routes.append(get_turn_route((grid_num, 0), (1, 2)))
        routes.append(get_turn_route((1, grid_num + 1), (3, 0)))

        # turn right
        #routes.append(get_turn_route((grid_num + 1, 1), (2, 1)))
        #routes.append(get_turn_route((grid_num, grid_num + 1), (3, 2)))
        #routes.append(get_turn_route((1, 0), (1, 0)))
        #routes.append(get_turn_route((0, grid_num), (0, 3)))


    return routes


if __name__ == '__main__':
    args = parse_args()
    if args.roadnetFile is None:
        args.roadnetFile = "roadnet_%d_%d.json" % (args.gridNum, args.gridNum)
    if args.flowFile is None:
        args.flowFile = "flow_%d_%d%s.json" % (args.gridNum, args.gridNum, "_turn" if args.turn else "")

    grid = {
        "rowNumber": args.gridNum,
        "columnNumber": args.gridNum,
        "rowDistances": [args.rowDistance] * (args.gridNum - 1),
        "columnDistances": [args.columnDistance] * (args.gridNum - 1),
        "outRowDistance": args.rowDistance,
        "outColumnDistance": args.columnDistance,
        "intersectionWidths": [[args.intersectionWidth] * args.gridNum] * args.gridNum,
        "numLeftLanes": args.numLeftLanes,
        "numStraightLanes": args.numStraightLanes,
        "numRightLanes": args.numRightLanes,
        "laneMaxSpeed": args.laneMaxSpeed,
        "tlPlan": args.tlPlan
    }

    vehicle_template = {
        "length": args.vehLen,
        "width": args.vehWidth,
        "maxPosAcc": args.vehMaxPosAcc,
        "maxNegAcc": args.vehMaxNegAcc,
        "usualPosAcc": args.vehUsualPosAcc,
        "usualNegAcc": args.vehUsualNegAcc,
        "minGap": args.vehMinGap,
        "maxSpeed": args.vehMaxSpeed,
        "headwayTime": args.vehHeadwayTime
    }
    routes = generate_route(args.gridNum, args.turn)
    flow = []
    for route in routes:
        flow.append({
            "vehicle": vehicle_template,
            "route": route,
            "interval": args.interval
        })
    json.dump(flow, open(args.dir + args.flowFile, "w"), indent=2)