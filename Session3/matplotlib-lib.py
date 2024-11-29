# In the name of God

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y1 = ['a', 'b', 'c', 'd', 'e']
y2 = ['F', 'G', 'H', 'I', 'J']


plt.plot(x, y1, marker='o', color='black', linestyle='--', label='y1 data')
plt.plot(x, y2, marker='o', color='green', linestyle='-.', label='y2 data')


plt.xlabel('Numbers')
plt.ylabel('Alphabet')
plt.title('alpha-numeric plot', loc='center')
# plt.annotate('midle value', xy=(3,'c'), xytext=(3,4), arrowprops=dict(facecolor='red'))

# set background:
plt.grid(True, linestyle='--', linewidth=0.5, color='red')
plt.gca().set_facecolor('lightblue')
#image = plt.imread('/home/farhad/Desktop/data/Photo/Tablokhat/photo_2016-05-31_23-50-26.jpg')
#plt.imshow(image)

# add note:
plt.text(3, -3, 'this is a test plot', fontsize=10, color='red')
plt.legend()
plt.savefig('/home/farhad/Desktop/plt.pdf', dpi=300)
plt.show()



