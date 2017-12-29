import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randint(1,9,(6,4)),index=list("abcdef"),columns=list("ABCD"))
print(df)
cov = df.cov()
print(cov)
triu = np.triu(cov.values)
print(triu)
list = [x for x in triu.reshape(1,-1)[0] if x != 0]
print(list)