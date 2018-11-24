import matplotlib.pyplot as plt 
import numpy as np
import sys

def float_convert(x):
	return float(x)

print sys.argv[1]

for i in range(int(sys.argv[1])):

	with open(sys.argv[i+2]) as textFile:
		data1 = [list(map(float_convert, line.split())) for line in textFile]

	data1=np.array(data1)
	ss = str(sys.argv[i+2])
	labind = ss.rfind('-')+1
	lastind = ss.rfind('.')
	plt.plot(data1[:,0], data1[:,1], label="d = "+ss[labind:lastind])

# plt.yaxis.tick_right()
# print data[:,0]
# a = plt.plot(data1[:,1], data1[:,0]/1e5, label=sys.argv[1])
# b = plt.plot(data2[:,1], data2[:,0]/1e5, label=sys.argv[2])
# b = plt.plot(data3[:,1], data3[:,0]/1e5, label=sys.argv[3])
plt.legend(loc = "lower right")
plt.xlabel("Episodes")
plt.ylabel("Average reward per episode")
# plt.legend([sys.argv[2]])
# a.yaxis.tick_right()
plt.show()