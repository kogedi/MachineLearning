import monkdata as m
import dtree as dt

#Assignment 1
entropy = dt.entropy(m.monk3)
print("entropy", entropy)

#Assignment 3
print("Information Gain Monk 1")
for i in range(0,6):
    averageGain = dt.averageGain(m.monk1, m.attributes[i])
    print("Attribute", i+1,": ", averageGain)

print ("Information Gain Monk 2")
for i in range(0,6):
    averageGain = dt.averageGain(m.monk2, m.attributes[i])
    print("Attribute", i+1,": ", averageGain)

print("information Gain Monk 3")
for i in range(0,6):
    averageGain = dt.averageGain(m.monk3, m.attributes[i])
    print("Attribute", i+1,": ", averageGain)
    
