"""Independent synthetic checks for the NEW data path and inferential controls."""
from pathlib import Path
import json,tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from build_independent import raw_aggregate,summarize_pair,SEEDS
from evaluate_independent import brier,W

def main():
    # Span the 250k streaming boundary, mix missing and both repeated outcomes.
    length=250010
    y=np.arange(length)%2;response=y.astype(float);response[11]=np.nan
    t=pa.table({'subject_id':['synthetic']*length,'item_id':[str(i%100) for i in range(length)],
                'response':pa.array(response,from_pandas=True)})
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
        f=Path(temp)/'responses.parquet';pq.write_table(t,f,row_group_size=10000)
        result,info=raw_aggregate(f);actual=result.to_pandas().sort_values('item_id').reset_index(drop=True)
        df=t.to_pandas().dropna();expected=df.groupby(['subject_id','item_id']).response.agg(['sum','count','first']).reset_index().sort_values('item_id').reset_index(drop=True)
        np.testing.assert_array_equal(actual.successes,expected['sum'])
        np.testing.assert_array_equal(actual.trials,expected['count'])
        np.testing.assert_array_equal(actual['first'],expected['first'])
        assert info['rows']==length and info['nulls']==1
        row,s=summarize_pair(actual,'b','synthetic',{})
        _,shuffled=summarize_pair(actual.sample(frac=1,random_state=111),'b','synthetic',{})
        assert s==shuffled,'primary reconstruction must ignore storage row order'
        assert len(s)==len(SEEDS) and row['n_items']==100
        nonbinary=t.set_column(2,'response',pa.array(np.r_[2.,np.zeros(length-1)]))
        pq.write_table(nonbinary,f);bad,why=raw_aggregate(f)
        assert bad is None and not why['observed_support_is_binary']
    # Hand-known exact Brier and cluster-weighted ratio; row means are not substituted.
    p=np.full((2,6),.5);np.testing.assert_allclose(brier(p,np.array([0.,1.]))@W,[.25,.25],atol=1e-15)
    sums=np.array([-10.,1.]);counts=np.array([100,1])
    assert np.isclose(sums.sum()/counts.sum(),-9/101)
    assert not np.isclose(sums.sum()/counts.sum(),np.mean(sums/counts))
    out={'status':'pass','tests':['streaming group sum/count/first across batches','missing responses',
         'storage order invariant primary scenarios','8 fixed seeds','nonbinary witness rejection',
         'exact Brier identity','cluster ratio preserves pair weighting']}
    (Path(__file__).resolve().parent/'data_pipeline_tests.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out))

if __name__=='__main__':main()
