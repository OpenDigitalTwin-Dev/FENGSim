#!BPY

"""
Name: 'Structure Synth Object (.gss)'
Blender: 244
Group: 'Import'
Tooltip: 'Load a Structure Synth Object'
"""

__author__ = "David Bucciarelli"

# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
#
# ***** END GPL LICENCE BLOCK *****

import Blender
from Blender import *
from Blender.Mathutils import *

MaterialList = []

def handleColor(obj,colR,colG,colB):
	matName = "str" + str(colR) + str(colG) + str(colB)

	materialmatch = 0

	for mat in Material.Get():
#		print "mat: " + mat.name
#		print "matName: " + matName
		if mat.name == matName:
			print "match"
			materialmatch = mat

	if materialmatch == 0:
			mat= Blender.Material.New(matName)
			mat.setRGBCol(colR, colG, colB)
			MaterialList.append(matName)
			materialmatch = mat

			me = obj.getData()
	
			me.addMaterial(mat)
	
			me.update()
	index = 0
	for name in MaterialList:
#		print "itr: " + name
#		print "search: " + materialmatch.name
		if materialmatch.name == name:
			print "matches"
			return index
		index +=1

	return 0


#================================ 
def ImportFunction(fileName):
#================================ 
	editmode = Window.EditMode()
	if editmode: Window.EditMode(0)

	print "Reading generic Structure Synth object: " + fileName
	file = open(fileName, "rb")

	scene = Scene.GetCurrent()

	
	verts = [ [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1] ]
	faces = [ [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 4, 7], [0, 7, 1], [1, 7, 6], [1, 6, 2], [2, 6, 5], [2, 5, 3], [4, 0, 3], [4, 3, 5] ]


	objCount = 0

	transVerts = []
	transVertNormals = []
	transFaces = []
	transFaceSmooth = []
	transFaceMat = []

	mesh = Mesh.New()
	obj = Blender.Object.New('Mesh',"struct")
	obj.link(mesh)
	scene.objects.link(obj)

	for line in file:



		if objCount % 2500 == 0:
			print "Import progress report: " + str(objCount) + " objects"

		args = line.split()
		argsIndex = 0

		colR = 1
		colB = 0
		colG = 1

		if args[argsIndex] == "s":
			argsIndex += 1
			cx = eval(args[argsIndex])
			argsIndex += 1
			cy = eval(args[argsIndex])
			argsIndex += 1
			cz = eval(args[argsIndex])
			argsIndex += 1
			radius = eval(args[argsIndex])
			argsIndex += 1

			colR = eval(args[argsIndex])
			argsIndex += 1
			colG = eval(args[argsIndex])
			argsIndex += 1
			colB = eval(args[argsIndex])
			argsIndex += 1

			matindex = handleColor(obj,colR,colG,colB)

			meshSph = Mesh.Primitives.Icosphere(2, 2.0 * radius)

			transMatrix = TranslationMatrix(Vector([cx, cy, cz]))

			offset = len(transVerts)
			for f in meshSph.faces:
				transFaces.append([f.verts[0].index + offset, f.verts[1].index + offset, f.verts[2].index + offset])
				transFaceSmooth.append(True)
				transFaceMat.append(matindex)

			for v in meshSph.verts:
				transVerts.append(v.co * transMatrix)
				transVertNormals.append(v.no)


		elif args[argsIndex] == "m":
			argsIndex += 1;
			
			existingvertcount = len(transVerts)
#			print "vertcount" + str(existingvertcount)
			
			startBase = Vector([args[argsIndex+0], args[argsIndex+1], args[argsIndex+2]])
			startTransMatrix = TranslationMatrix(startBase);
			
			
			
			startDir1 = Vector([args[argsIndex+4], args[argsIndex+5], args[argsIndex+6]])
			startDir2 = Vector([args[argsIndex+8], args[argsIndex+9], args[argsIndex+10]])
			
			endBase = Vector([args[argsIndex+12], args[argsIndex+13], args[argsIndex+14]])
			endTransMatrix = TranslationMatrix(endBase);
			
			endDir1 = Vector([args[argsIndex+16], args[argsIndex+17], args[argsIndex+18]])
			endDir2 = Vector([args[argsIndex+20], args[argsIndex+21], args[argsIndex+22]])
			
			end = endBase - startBase;
			
			transVerts.append(startBase);
			transVerts.append(startDir1*startTransMatrix);
			transVerts.append(startDir2*startTransMatrix);
			transVerts.append((startDir1+startDir2)*startTransMatrix);
			transVerts.append((end+endDir1)*startTransMatrix);
			transVerts.append((end+endDir2)*startTransMatrix);
			transVerts.append((end+endDir1+endDir2)*startTransMatrix);
			transVerts.append(end*startTransMatrix);
						
#			transFaces.append([existingvertcount+0, existingvertcount+1, existingvertcount+3,existingvertcount+2])
			transFaces.append([existingvertcount+0, existingvertcount+1, existingvertcount+4,existingvertcount+7])
			transFaces.append([existingvertcount+0, existingvertcount+2, existingvertcount+5,existingvertcount+7])
			transFaces.append([existingvertcount+1, existingvertcount+3, existingvertcount+6,existingvertcount+4])
			transFaces.append([existingvertcount+2, existingvertcount+3, existingvertcount+6,existingvertcount+5])
#			transFaces.append([existingvertcount+7, existingvertcount+4, existingvertcount+6,existingvertcount+5])
			
			argsIndex = 25

			colR = eval(args[argsIndex+0])
			colG = eval(args[argsIndex+1])
			colB = eval(args[argsIndex+2])
			
			matindex = handleColor(obj,colR,colG,colB)
			
			transFaceMat.append(matindex)
			transFaceMat.append(matindex)
			transFaceMat.append(matindex)
			transFaceMat.append(matindex)
			
			

		elif args[argsIndex] == "b":
			argsIndex += 1
			transMatrix = Matrix([args[argsIndex + 0],  args[argsIndex + 1],  args[argsIndex + 2],  args[argsIndex + 3]],
								 [args[argsIndex + 4],  args[argsIndex + 5],  args[argsIndex + 6],  args[argsIndex + 7]],
								 [args[argsIndex + 8],  args[argsIndex + 9],  args[argsIndex + 10], args[argsIndex + 11]],
								 [args[argsIndex + 12], args[argsIndex + 13], args[argsIndex + 14], args[argsIndex + 15]])
			argsIndex += 16

			colR = eval(args[argsIndex])
			argsIndex += 1
			colG = eval(args[argsIndex])
			argsIndex += 1
			colB = eval(args[argsIndex])
			argsIndex += 1

			matindex = handleColor(obj,colR,colG,colB)

			offset = len(transVerts)
			for f in faces:
				transFaces.append([f[0] + offset, f[1] + offset, f[2] + offset])
				transFaceSmooth.append(False)
				transFaceMat.append(matindex)

			for v in verts:
				transVerts.append(Vector(v) * transMatrix)
				transVertNormals.append([])
		else:
			print "Unknown primitive type: " + args[argsIndex]


		objCount += 1

	mesh.verts.extend(transVerts)
	mesh.faces.extend(transFaces)	

	print len(transFaceMat)

	if(len(transFaceMat)>0):	
		index = 0
		for f in mesh.faces:
			if transFaceMat[index]:
#				print "index: " + str(index) + " facemat:" + str(transFaceMat[index])
				f.mat = transFaceMat[index]
			index += 1
	
	if(len(transFaceSmooth)>0):	
		index = 0
		for f in mesh.faces:
			if transFaceSmooth[index]:
				f.smooth = 1
			index += 1
	
	if(len(transVertNormals)>0):
		index = 0
		for v in mesh.verts:
				if len(transVertNormals[index]) > 0:
					v.no = transVertNormals[index]
					index += 1

	print "Total object count: " + str(objCount)

	file.close()
	Window.RedrawAll()
	print "Done"

#================================
# Main
#================================
Blender.Window.FileSelector(ImportFunction, "Import Structure Synth Object")
