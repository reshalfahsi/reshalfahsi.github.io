function meme(){

var myimages=new Array()
myimages=["https://i.ibb.co/PmBFVg3/aiengineermeme.png", "https://i.ibb.co/VNQVR9X/jian-yang-hot-dog-gif.gif", "https://media.giphy.com/media/xT9DPp7lYtKlM0QzII/giphy.gif", "https://miro.medium.com/max/1564/1*hBccbACZZkXyRl_YI1abFQ.jpeg", "https://images.ctfassets.net/be04ylp8y0qc/1AYDDqzqoLNU7GyvMOLlTL/146ab0e8b96a25c1b3538a2929dae646/meme_6cce0f3bba63820b84f52ce05a822975_800.png?fm=jpg"
]

var ry=Math.floor(Math.random()*myimages.length)

if (ry>=myimages.length)
	ry = myimages.length - 1; 
	document.write('<img src="'+myimages[ry]+'" width="500" height="250" style="width:50%; height:25%;">')
}

meme()
