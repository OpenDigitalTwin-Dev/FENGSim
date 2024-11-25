#javascript

function pad(number) {
	number = number + ''; // convert to string
 	while (number.length < 4) {
		number = "0" + number;
	}
	return number;
}


Builder.load("NouveauMovie.es");
max =1000 ;
for (i = 0; i <=  max; i+=1) {
       c = i/max;

	Builder.reset();	
	Builder.setSize(0,100);
	Builder.define("_rz",c*360);
	Builder.define("_md",20+c*3000);
	Builder.define("_dofa",0.2+ 0.1*Math.sin(c*3.1415*2));

	Builder.build();

       // ---- Sunflow raytrace -----
       /*
       name = "f:/Test/out" + pad(i);
	Builder.templateRenderToFile("Sunflow-Colored.rendertemplate", name + ".sc",true); 
       Builder.execute('"C:/Program Files/Java/jdk1.6.0_21/bin/java"', '-Xmx1G -server -jar  "%SUNFLOW%/sunflow.jar" ' +  name + ".sc -nogui -o " + name + ".png", true);
   	*/

	 // ---- Internal raytrace ------
       Builder.raytraceToFile("N" + pad(i) + ".png",true);
}
