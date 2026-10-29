"""Export a NEW post-LSA FG5% single bank; never modify its source."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-bank',type=Path,required=True)
    p.add_argument('--output-bank',type=Path,required=True)
    a=p.parse_args()
    from src.semantic_bank_filter import export_filtered_bank
    report=export_filtered_bank(a.source_bank,a.output_bank)
    print('LSA_POSTFILTER_COMPLETE',report['total_kept'])

if __name__=='__main__':main()
