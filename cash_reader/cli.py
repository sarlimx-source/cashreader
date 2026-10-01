import argparse
import json

from .database import BillDatabase
from .pipeline import CashReader


def main():
    parser = argparse.ArgumentParser(description="U.S. cash reader")
    parser.add_argument("--image", help="Path to a bill image")
    parser.add_argument("--camera", type=int, help="Camera device index")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--db", default="data/bills.db", help="SQLite database path")
    parser.add_argument("--recent", type=int, metavar="N",
                        help="Show the N most recent saved scans")
    parser.add_argument("--find-serial", metavar="SERIAL",
                        help="Find saved scans with this serial number")
    args = parser.parse_args()

    db = BillDatabase(args.db)

    if args.recent is not None:
        print(json.dumps(db.recent(args.recent), indent=2))
        return

    if args.find_serial:
        print(json.dumps(db.find_by_serial(args.find_serial), indent=2))
        return

    if not args.image and args.camera is None:
        parser.error("Choose --image, --camera, --recent, or --find-serial")

    reader = CashReader(database=db)

    if args.image:
        result = reader.scan_image(args.image, save_debug=args.save_debug)
        print(json.dumps(result, indent=2))
        return

    result = reader.scan_camera(args.camera, save_debug=args.save_debug)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
