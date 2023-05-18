import visualkeras
from model import selectModel



from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 20)


model = selectModel('resNetModel',d1=256,d2=256,channels=3,outputs=50,blocks=3)

# visualkeras.layered_view(model, legend=True, font=font).show()

counter = {}
for layer in model.layers:
    tp = layer.__class__.__name__
    if tp in counter.keys():
        counter[tp] +=1
    else:
        counter[tp] = 1


print(counter)