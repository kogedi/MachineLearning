import monkdata as m
import dtree as d
import PyQt5 as pq
import drawtree_qt5 as drqt5
import numpy as np

# #Assignment 1
# entropy = d.entropy(m.monk3)
# print("entropy", entropy)

# #Assignment 3
# print("Information Gain Monk 1")
# for i in range(0,6):
#     averageGain = d.averageGain(m.monk1, m.attributes[i])
#     print("Attribute", i+1,": ", averageGain)

# print ("Information Gain Monk 2")
# for i in range(0,6):
#     averageGain = d.averageGain(m.monk2, m.attributes[i])
#     print("Attribute", i+1,": ", averageGain)

# print("information Gain Monk 3")
# for i in range(0,6):
#     averageGain = d.averageGain(m.monk3, m.attributes[i])
#     print("Attribute", i+1,": ", averageGain)
    
    
# #Assignment 5
#t1 = d.buildTree(m.monk1, m.attributes, 10);
# t2 = d.buildTree(m.monk2, m.attributes);
# t3 = d.buildTree(m.monk3, m.attributes);


# # Check(): Measure fraction of correctly classified samples
# print("Check Monk 1",d.check(t1, m.monk1))
# print("Check Monk 1 Test",d.check(t1, m.monk1test))
# print("Check Monk 2",d.check(t2, m.monk2))
# print("Check Monk 2 Test",d.check(t2, m.monk2test))
# print("Check Monk 3",d.check(t3, m.monk3))
# print("Check Monk 3 Test",d.check(t3, m.monk3test))

#Assignment 6
#t1 = []
#
# t1.insert(1, A2(A5(+++-)A5(+A1(+A4(+-+)+)+-)A4(A5(--+A1(--+))--)))
#drqt5.drawTree(t1)

#Assignment 7

#print("Tree Monk 3", t3)
# print("Check Monk 3",d.check(t3, m.monk3))
# print("Check Monk 3 Test",d.check(t3, m.monk3test))
# print("Start pruning")

#Devividing the data into Training and validation data
import random

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#fraction = [0.8, 0.8]
entireRuns = 50
N = 0
accuracyfraction = np.zeros((len(fraction), int(entireRuns)))
for N in range(entireRuns):
    for fracind in range(len(fraction)):
        print("For fraction of ", fraction[fracind])
        
        monk1train, monk1val = partition(m.monk1, fraction[fracind])
        monk3train, monk3val = partition(m.monk3, fraction[fracind])
        # Built up the tree of Monk 3 used since it's the simplest one.
        t3 = d.buildTree(monk3train, m.attributes)
        t1 = d.buildTree(monk1train, m.attributes)
        PrunChances = 0
        PrunChances= d.allPruned(t3)
        j = 0
        maxaccuracy = []
        repeat = True
        while repeat == True:
            accuracy = []
            
            for k in range(len(PrunChances)):
                #print("Check",d.check(PrunChances[k], monk3val))
                accuracy.insert(k, round(d.check(PrunChances[k], monk3val), 5))
                #print(k,":",accuracy[k], "  for tree:", PrunChances[k])
            
                'Determin the best subset in terms of accuracy'
                if accuracy[k] == 0:
                    accuracy.insert(k, 0.0000001)
            maxaccuracy.insert(j, max(accuracy)) 
            #print("maxaccuracy[j]", maxaccuracy[j]) 
            maxaccurayID = accuracy.index(maxaccuracy[j])
            t3sub = PrunChances[maxaccurayID]
            print("Best subset is ",maxaccurayID,"with an accuracy of",maxaccuracy[j])
            print(t3sub)
            #drqt5.drawTree(t3sub)
            print(" ")
            # Prune the subset again
            PrunChances = 0
            PrunChances = d.allPruned(t3sub)
            if len(PrunChances) < 2:
                break
                
            if len(maxaccuracy) > 1:
                #print("len", len(maxaccuracy))
                if maxaccuracy[j] <= maxaccuracy[j-1]:
                    repeat = False
                    print("Rejected. Best Dataset beforecls")
            
            accuracyfraction[fracind,N] = maxaccuracy[j]
            j += 1
    #maxaccuracyfraction = max(accuracyfraction)
    #maxaccuracyfractionID = accuracyfraction.index(maxaccuracyfraction)
    #print("The highest accuracy is generated with a fraction of",fraction[maxaccuracyfractionID], " w/ value",maxaccuracyfraction )
    
#Datenaufbereitung
print("accuracyfraction", accuracyfraction)
#Statistics
#mean = 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Erstelle sechs Boxplots
plt.figure(figsize=(10, 6))
x_values = []
for i, data in enumerate(accuracyfraction):
    plt.subplot(2, 3, i + 1)  # Subplots für jeden Boxplot
    mean = np.mean(data)
    x_values.insert(i, mean)
    std_dev = np.std(data)
    sns.boxplot(data=data,width=0.5)
    plt.title(f'Fraction: {fraction[i]}\nMean = {mean:.2f}, StdDev = {std_dev:.2f}')
    plt.xlabel('')
    plt.ylabel('Accuracy')

plt.tight_layout()  # Damit die Plots nicht überlappen
plt.show()

# Beispiel-Array mit 6 Werten
data = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# X-Werte als Fraktionen von 0 bis 1 erstellen 


# Plot erstellen
plt.plot(data, x_values, marker='o', linestyle='-')
plt.xlabel('Fraction')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Fraction')
plt.grid(True)  # Raster anzeigen, um die Werte besser abzulesen
plt.show()
