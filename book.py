import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import seaborn as sns
import matplotlib.pyplot as plt

# create a pandas dataframe from book.csv
data = pd.read_csv('book.csv')

# print(data.shape) # 2000 rows, 11 columns
# print(data.info()) # no null values
# print(data.columns)

# apriori algorithm
frequent_itemsets1 = apriori(data, min_support=0.02, use_colnames=True, max_len=5) # 2% min support
# print(frequent_itemsets1) # 271 itemsets

frequent_itemsets2 = apriori(data, min_support=0.05, use_colnames=True, max_len=4) # 5% min support
# print(frequent_itemsets2) # 100 itemsets

# Sorting based on support value 
ordered_frequent_itemsets2 = frequent_itemsets2.sort_values('support', ascending=False)
print("Frequent itemsets in descending order: \n", ordered_frequent_itemsets2)

# Visualizing top 10 itemsets having highest support value
plt.bar(x = list(range(0,10)),height = ordered_frequent_itemsets2.support[0:10],color='r');plt.xticks(list(range(0,10)),ordered_frequent_itemsets2.itemsets[0:10])
plt.xlabel('item-sets');plt.ylabel('support')
plt.show()
# Cookbooks are the most frequent itemset


# Association Rule Mining based on confidence using frequent itemsets that has min support of 5%
res_conf_1 = association_rules(frequent_itemsets2, metric='confidence', min_threshold=0.7)
# print(res_conf_1[['support', 'confidence', 'lift']])

res_conf_2 = association_rules(frequent_itemsets2, metric='confidence', min_threshold=1)
# print(res_conf_2[['support', 'confidence', 'lift']])

# Association Rule Mining based on lift using frequent itemsets that has min support of 5%
res_lift_2 = association_rules(frequent_itemsets2, metric='lift', min_threshold=1)
res_lift_2.sort_values('lift',ascending=False,inplace=True)
print(res_lift_2[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
sns.scatterplot(res_lift_2['support'], res_lift_2['confidence'], hue=res_lift_2['lift'])

plt.show()


########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))


ma_X = res_lift_2.antecedents.apply(to_list)+res_lift_2.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy  = res_lift_2.iloc[index_rules,:]
print(rules_no_redudancy)

# Sorting them with respect to list and getting top 10 rules 
print(rules_no_redudancy.sort_values('lift',ascending=False).head(10))


