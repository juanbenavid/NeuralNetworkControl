
import numpy as np

tankHead = np.array([            
74.50  ,         
74.34  ,         
74.21  ,         
74.10  ,         
74.20  ,         
74.37  ,         
74.58  ,         
74.70  ,         
74.86  ,         
75.09  ,         
75.14  ,         
75.08  ,         
75.20  ,         
75.35  ,         
75.53  ,         
75.69  ,         
75.86  ,         
76.00  ,         
75.90  ,         
75.70  ,         
75.53  ,         
75.47  ,         
75.46  ,         
75.49  ,         
75.50  ,         
75.47  ,         
75.45  ,         
75.52  ,         
75.62  ,         
75.60  ,         
75.69  ,         
75.75  ,         
75.92  ,         
75.93  ,         
75.51  ,         
75.01  ,         
74.56  ,         
74.11  ,         
73.68  ,         
73.27  ,         
72.86  ,         
72.54  ,         
72.61  ,         
72.88  ,         
73.29  ,         
73.65  ,         
73.91  ,         
74.10  ,         
74.31  ,         
74.50  ,         
74.62  ,         
74.49  ,         
74.46  ,         
74.78  ,         
75.09  ,         
75.24  ,         
75.37  ,         
75.47  ,         
75.57  ,         
75.65  ,         
75.87  ,         
75.96  ,         
75.83  ,         
75.71  ,         
75.62  ,         
75.33  ,         
74.83  ,         
74.56  ,         
74.20  ,         
73.86  ,         
73.46  ,         
73.01  ,         
72.54  ,         
72.41  ,         
72.41  ,         
72.48  ,         
72.61  ,         
72.84  ,         
73.08  ,         
73.38  ,         
73.52  ,         
73.58  ,         
73.53  ,         
73.41  ,         
73.32  ,         
73.24  ,         
73.11  ,         
73.01  ,         
72.94  ,         
72.88  ,         
73.05  ,         
73.24  ,         
73.53  ,         
73.87  ,         
74.10  ,         
74.31  ,         
74.50  ,         
74.74  ,         
75.04  ,         
75.10  ,         
75.16  ,         
75.45  ,         
75.63  ,         
75.69  ,         
75.72  ,         
75.60  ,         
75.39  ,         
75.22  ,         
75.09  ,         
75.15  ,         
75.26  ,         
75.28  ,         
75.60  ,         
75.96  ,         
76.05  ,         
76.08  ,         
75.96  ,         
75.74  ,         
75.55  ,         
75.34  ,         
74.92  ,         
74.41  ,         
73.96  ,         
73.62  ,         
73.42  ,         
73.28  ,         
73.18  ,         
73.15  ,         
73.08  ,         
72.88  ,         
72.55  ,         
72.48  ,         
72.42  ,         
72.34  ,         
72.27  ,         
72.21  ,         
72.22  ,         
72.33  ,         
72.43  ,         
72.55  ,         
72.62  ,         
72.61  ,         
72.45  ,         
72.30  ,         
72.20  ,         
72.12  ,         
72.16  ,         
72.30  ,         
72.55  ,         
72.81  ,         
73.12  ,         
73.59  ,         
74.17  ,         
74.65  ,         
74.82  ,         
74.98  ,         
75.20  ,         
75.38  ,         
75.58  ,         
75.37  ,         
75.20  ,         
75.15  ,         
75.19  ,         
75.32  ,         
75.31  ,         
75.17  ,         
74.97  ,         
74.72  ,         
74.50  ,         
74.67  ,           
74.56])           

tankFlow = np.array([          
-34.56 ,         
-27.78 ,         
-22.91 ,         
20.17  ,         
36.95  ,         
45.83  ,         
24.63  ,         
34.77  ,         
49.99  ,         
10.81  ,         
-13.87 ,         
26.86  ,         
31.75  ,         
38.94  ,         
33.99  ,         
36.71  ,         
28.65  ,         
-21.52 ,         
-41.82 ,         
-36.14 ,         
-14.03 ,         
-1.11  ,         
6.15   ,         
1.89   ,         
-7.19  ,         
-2.33  ,         
13.60  ,         
22.93  ,         
-4.34  ,         
18.48  ,         
12.36  ,         
37.63  ,         
23.44  ,         
-90.48 ,         
-106.41,         
-97.41 ,         
-96.21 ,         
-91.81 ,         
-88.30 ,         
-86.09 ,         
-69.48 ,         
-46.66 ,         
57.79  ,         
88.85  ,         
77.80  ,         
55.80  ,         
40.22  ,         
44.53  ,         
40.93  ,         
26.98  ,         
-29.18 ,         
-5.73  ,         
67.86  ,         
67.66  ,         
31.20  ,         
27.54  ,         
22.95  ,         
21.27  ,         
17.13  ,         
46.47  ,         
42.87  ,         
-27.62 ,         
-25.84 ,         
-19.45 ,         
-61.45 ,         
-105.76,         
-58.27 ,         
-78.48 ,         
-72.32 ,         
-84.64 ,         
-98.11 ,         
-98.92 ,         
-96.06 ,         
0.75   ,         
14.98  ,         
26.56  ,         
50.49  ,         
51.52  ,         
63.23  ,         
29.97  ,         
13.32  ,         
-9.77  ,         
-26.60 ,         
-18.59 ,         
-18.60 ,         
-28.05 ,         
-19.81 ,         
-15.69 ,         
-12.39 ,         
36.74  ,         
39.87  ,         
62.30  ,         
73.25  ,         
48.12  ,         
46.07  ,         
39.59  ,         
53.31  ,         
63.01  ,         
13.98  ,         
12.45  ,         
61.19  ,         
39.72  ,         
12.94  ,         
5.47   ,         
-24.44 ,         
-44.79 ,         
-37.14 ,         
-27.38 ,         
11.41  ,         
24.84  ,         
4.10   ,         
67.48  ,         
76.99  ,         
81.35  ,         
6.26   ,         
-25.80 ,         
-48.70 ,         
-40.40 ,         
-43.69 ,         
-90.49 ,         
-110.42,         
-94.65 ,         
-74.10 ,         
-42.08 ,         
-30.52 ,         
-20.22 ,         
-8.31  ,         
-12.98 ,         
-44.31 ,         
-70.81 ,         
-80.29 ,         
-12.03 ,         
-17.33 ,         
-14.39 ,         
-13.09 ,         
1.01   ,         
24.62  ,         
21.59  ,         
24.46  ,         
16.92  ,         
-4.08  ,         
-32.38 ,         
-32.28 ,         
-21.68 ,         
-17.29 ,         
8.20   ,         
30.31  ,         
52.54  ,         
56.11  ,         
66.09  ,         
101.57 ,         
125.07 ,         
101.71 ,         
36.71  ,         
35.09  ,         
47.23  ,         
38.97  ,         
41.19  ,         
-43.95 ,         
-35.90 ,         
-12.12 ,         
10.01  ,         
28.04  ,         
-2.27  ,         
-30.63 ,         
-43.48 ,         
-53.11 ,         
-48.30 ,         
36.89  ,         
-23.37 ,         
-24.90])      

Open = 1.0
Closed = 0.0

pump1 = np.array([              	                
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,            
Open])      

pump2 = np.array([               
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Closed,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open  ,          
Open ])      

pumpsFlow = np.array([     
191.08 ,         
190.44 ,         
190.33 ,         
188.63 ,         
187.08 ,         
186.91 ,         
187.53 ,         
187.11 ,         
187.33 ,         
189.00 ,         
189.53 ,         
186.19 ,         
185.85 ,         
184.56 ,         
184.81 ,         
184.07 ,         
184.67 ,         
187.74 ,         
189.76 ,         
189.68 ,         
189.99 ,         
188.83 ,         
188.11 ,         
187.83 ,         
187.06 ,         
186.98 ,         
186.39 ,         
185.53 ,         
186.36 ,         
186.40 ,         
187.23 ,         
185.83 ,         
186.63 ,         
115.35 ,         
116.83 ,         
116.91 ,         
117.51 ,         
117.88 ,         
118.28 ,         
118.91 ,         
118.67 ,         
118.49 ,         
191.61 ,         
188.26 ,         
186.96 ,         
187.00 ,         
187.84 ,         
186.82 ,         
186.83 ,         
187.38 ,         
189.85 ,         
189.04 ,         
185.59 ,         
184.72 ,         
186.76 ,         
185.97 ,         
186.73 ,         
188.24 ,         
186.90 ,         
182.89 ,         
182.90 ,         
112.78 ,         
112.88 ,         
112.76 ,         
114.51 ,         
116.75 ,         
115.74 ,         
117.03 ,         
117.30 ,         
118.34 ,         
119.51 ,         
120.33 ,         
120.12 ,         
193.39 ,         
193.57 ,         
191.75 ,         
189.85 ,         
190.32 ,         
189.54 ,         
191.85 ,         
192.12 ,         
192.45 ,         
194.04 ,         
193.20 ,         
193.80 ,         
194.88 ,         
194.18 ,         
194.00 ,         
193.95 ,         
191.75 ,         
192.46 ,         
188.36 ,         
187.03 ,         
188.32 ,         
186.63 ,         
187.04 ,         
184.74 ,         
182.83 ,         
186.83 ,         
186.80 ,         
184.48 ,         
185.49 ,         
186.01 ,         
186.88 ,         
190.27 ,         
191.24 ,         
189.86 ,         
189.09 ,         
187.02 ,         
186.29 ,         
187.68 ,         
183.03 ,         
181.65 ,         
180.15 ,         
112.03 ,         
112.99 ,         
113.97 ,         
113.83 ,         
114.28 ,         
116.14 ,         
117.52 ,         
117.16 ,         
117.05 ,         
116.31 ,         
116.17 ,         
116.12 ,         
116.44 ,         
116.86 ,         
117.60 ,         
119.23 ,         
119.66 ,         
194.95 ,         
195.45 ,         
195.18 ,         
195.15 ,         
195.12 ,         
194.51 ,         
194.10 ,         
193.57 ,         
194.30 ,         
195.09 ,         
197.00 ,         
197.32 ,         
196.29 ,         
196.08 ,         
193.52 ,         
192.11 ,         
191.50 ,         
190.61 ,         
188.47 ,         
183.78 ,         
181.94 ,         
182.66 ,         
186.53 ,         
186.33 ,         
184.35 ,         
184.99 ,         
184.30 ,         
189.04 ,         
189.26 ,         
188.90 ,         
188.87 ,         
187.72 ,         
189.34 ,         
190.02 ,         
191.33 ,         
193.18 ,         
193.14 ,         
186.33 ,         
189.47 ,         
189.65])


demandFlow = pumpsFlow - tankFlow

demandFrac = demandFlow / np.max(demandFlow)

"""
returns time series of tankFlow,tankHead,pump1 status, pump2 status, demandFlow and demandFrac
from minitown epanet simulation over 170 hour period. tuple of vectors 
(tankFlow,tankHead,pump1,pump2,pumpsFlow,demandFlow,demandFrac)
"""
def minitownData():
    return (tankFlow,tankHead,pump1,pump2,pumpsFlow,demandFlow,demandFrac)

def minitownTrain():
    X_train = np.array([ np.array([p1,p2,tH,dem]) for p1,p2,tH,dem in zip(pump1,pump2,tankHead,demandFlow) ])
    Y_train = tankFlow
    return X_train,Y_train


