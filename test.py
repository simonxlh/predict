import pandas as pd
import numpy as np
df = pd.DataFrame(index=list(x for x in range(1,10,2)))
_data = [[3,10],[5,13]]
_cpu_df = pd.DataFrame(data=_data, columns=('time', 'value'))
print(_cpu_df)
# df['work'] = np.nan
# print(df)