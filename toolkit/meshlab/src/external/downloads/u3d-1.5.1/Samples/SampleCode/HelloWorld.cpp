//***************************************************************************
//
//  Copyright (c) 1999 - 2006 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//***************************************************************************

/**
	@file HelloWorld.cpp

	This is sample application to demonstrate all necessary steps needed to
	create basic scene with textured animated cube and write it to file.

	At high level overview the steps are like following:
	1. Initialization and aquiring some global objects (like CoreServices).
	2. Create camera and its associated resource.
	3. Create point light and its associated resource.
	4. Create model node and link it to cube CLOD mesh.
	5. Add shading modifier to model with material and texture resources.
	6. Add animation modifier to model.
	7. Add CLOD modifier to model.
	8. Write scene to 'output.u3d' file.
	9. Uninitialization.

	This sample should be linked to IFXCoreStatic.lib which in turn manages
	dynamic linkage to IFXCore.dll and IFXExporting.dll (in plugins directory).
*/

// This definition should be used before inclusion of all headers, so all of
// the IFXGUIDs are actually defined in this file. This should be done only
// in single .cpp file to avoid linkage conflicts.
#define IFX_INIT_GUID

#include "IFXCoreCIDs.h"
#include "IFXSceneGraph.h"
#include "IFXView.h"
#include "IFXLight.h"
#include "IFXNode.h"
#include "IFXUnknown.h"
#include "IFXCoreServices.h"
#include "IFXGUID.h"
#include "IFXCOM.h"
#include "IFXAuthorMesh.h"
#include "IFXAuthorCLODResource.h"
#include "IFXWriteManager.h"
#include "IFXExportingCIDs.h"
#include "IFXStdio.h"
#include "IFXShaderLitTexture.h"
#include "IFXShadingModifier.h"
#include "IFXModifierChain.h"
#include "IFXMaterialResource.h"
#include "IFXTextureObject.h"
#include "IFXAnimationModifier.h"
#include "IFXMixerConstruct.h"
#include "IFXMotionResource.h"
#include "IFXCLODModifier.h"
#include "IFXAuthorGeomCompiler.h"


// The main functions that creates scenegraph and writes it to file
IFXRESULT CreateScene();
// This functions creates a cube mesh
IFXRESULT CreateCubeMesh(IFXAuthorCLODMesh* pClodMesh);

// Entry point
int main()
{
	IFXRESULT result = IFX_OK;

	// Initialization. Must be done before any usage of the library.
	result = IFXCOMInitialize();

	if (IFXSUCCESS(result))
		result = CreateScene();

	// Uninitialize IFXCOM
	if (IFXSUCCESS(result))
		result = IFXCOMUninitialize();

	return result;
}


IFXRESULT CreateScene()
{
	IFXRESULT result = IFX_OK;
	IFXCoreServices* pCoreServices = NULL;
	IFXNode* pWorldNode = NULL;
	IFXSceneGraph* pSceneGraph = NULL;

	/*********************************************************************************
	*
	*		Scene initialization
	*
	*********************************************************************************/

	// Create Core services instance.
	// This object is the core of the entire system.
	// When created, the CoreServices object will create a Scenegraph,
	// as well as several other management objects.
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXCoreServices, IID_IFXCoreServices,
									(void**)&pCoreServices);

	// Initialisation of an IFXCoreServices object.
	// Must be done before any usage of the pCoreServices.
	if (IFXSUCCESS(result))
		result = pCoreServices->Initialize(IFXPROFILE_BASE);

	// Get a handle to the scenegraph from the newly created CoreServices object.
	// The scenegraph contains all the information about the scene.
	if (IFXSUCCESS(result))
		result = pCoreServices->GetSceneGraph(IID_IFXSceneGraph, (void**)&pSceneGraph);

	/*********************************************************************************
	*
	*			Camera initialization
	*
	*********************************************************************************/

	IFXString viewNodeName("Camera1");
	U32 viewNodeID = 0;
	IFXString viewResourceName("CameraResource1");
	U32 viewResourceID = 0;
	IFXPalette* pNodePalette = NULL;
	IFXView* pViewNode = NULL;
	IFXViewResource* pViewResource = NULL;
	IFXPalette* pViewPalette = NULL;
	U32 worldNodeID = 0;

	// Viewport settings
	F32 viewportWidth = 500;
	F32 viewportHeight = 500;
	F32 viewProjection = 34.515877f;

	// Create View node
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXView, IID_IFXView, (void**)&pViewNode);

	// Initialize the View Node object
	if (IFXSUCCESS(result))
		result = pViewNode->SetSceneGraph(pSceneGraph);

	if (IFXSUCCESS(result))
		result = pSceneGraph->GetPalette(IFXSceneGraph::NODE, &pNodePalette);

	// Retrieve the World node from the Node Hierarchy Palette
	// and use it as the parent node of the view.
	if (IFXSUCCESS(result))
		result = pNodePalette->GetResourcePtr(0, IID_IFXNode, (void**)&pWorldNode);

	// Set the World to be the parent of the View Node
	if (IFXSUCCESS(result))
		result = pViewNode->AddParent(pWorldNode);

	// Add an entry for the view node to the node palette
	if (IFXSUCCESS(result))
		result = pNodePalette->Add(viewNodeName, &viewNodeID);

	// Point the node palette entry for the view to the view component
	if (IFXSUCCESS(result))
		result = pNodePalette->SetResourcePtr(viewNodeID, pViewNode);

	// Set the view node's local transtalion matrix
	if (IFXSUCCESS(result))
	{
		IFXMatrix4x4 matrix;
		matrix.Reset();
		matrix.Set(
					IFXVector3(1.000000f, 0.000000f, 0.000000f),
					IFXVector3(0.000000f, 0.258819f, 0.965926f),
					IFXVector3(0.000000f, -0.965926f, 0.258819f),
					IFXVector3(0.000000f, -241.481461f, 64.704765f)
				  );
		result = pViewNode->SetMatrix(0, &matrix);
	}

	// Create View resource
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXViewResource, IID_IFXViewResource,
									(void**)&pViewResource);

	// Init View Resource with IFXSceneGraph reference
	if (IFXSUCCESS(result))
		result = pViewResource->SetSceneGraph(pSceneGraph);

	// Get View Palette reference
	if (IFXSUCCESS(result))
		result = pSceneGraph->GetPalette(IFXSceneGraph::VIEW, &pViewPalette);

	// Add IFXViewResource to the View Palette
	if (IFXSUCCESS(result))
		result = pViewPalette->Add(viewResourceName, &viewResourceID);
	
	// Assign View resource to View node.
	if (IFXSUCCESS(result))
		result = pViewNode->SetViewResourceID(viewResourceID);
	
	// Initialize the new palette entry with the View Resource
	if (IFXSUCCESS(result))
		result = pViewPalette->SetResourcePtr(viewResourceID, pViewResource);

	// Set root node for this resource. The camera will only see those
	// nodes at or below the node in the parent/child hierarchy.
	if (IFXSUCCESS(result))
		result = pViewResource->SetRootNode(worldNodeID, 0);

	// Set viewport
	if (IFXSUCCESS(result))
	{	
		//Set View Port
		IFXF32Rect rect;
		rect.m_Width = viewportWidth;
		rect.m_Height = viewportHeight;
		result = pViewNode->SetViewport(rect);
	}

	// Set projection
	if (IFXSUCCESS(result))
		result = pViewNode->SetProjection(viewProjection);

	// 3-point perspective projection and screen position units in screen pixels
	if (IFXSUCCESS(result))
		pViewNode->SetAttributes(IFX_PERSPECTIVE3 | IFX_SCREENPIXELS); 

	// Release interfaces
	IFXRELEASE(pViewPalette)
	IFXRELEASE(pViewResource)
	IFXRELEASE(pViewNode)

	/*********************************************************************************
	*
	*			Light Setup
	*
	*********************************************************************************/

	IFXString lightResourceName("LightResource1");
	U32 lightResourceID = 0;
	IFXString lightNodeName("PointLight1");
	U32 lightNodeID = 0;
	IFXPalette* pLightPalette = NULL;
	IFXLightResource* pLightResource = NULL;
	IFXLight* pLightNode = NULL;

	// Light parameters
	IFXLightResource::LightType lightType = IFXLightResource::POINT;
	F32 lightAttenuation[3] = {0, 0, 0};
	IFXVector4 lightColor(1, 1, 1);
	F32 lightIntensity = 1.0f;

	// Create Light node
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXLight, IID_IFXLight, (void**)&pLightNode);
	
	// Initialize the light
	if (IFXSUCCESS(result))
		result = pLightNode->SetSceneGraph(pSceneGraph);

	// Set the parent of the light to be the World. 
	// As with all nodes, Light nodes can appear anywhere 
	// in the scene hierarchy.
	if (IFXSUCCESS(result))
		result = pLightNode->AddParent(pWorldNode);

	// Add an entry into the Node 
	// Hierarchy Palette for the new Light node.	
	if (IFXSUCCESS(result))
		result = pNodePalette->Add(lightNodeName, &lightNodeID);

	// Initialize the new palette entry with the Light node
	if (IFXSUCCESS(result))
		result = pNodePalette->SetResourcePtr(lightNodeID, pLightNode);

	// Set Light Node translation Matrix
	if (IFXSUCCESS(result))
	{
		IFXMatrix4x4 matrix;
		matrix.Reset();
		matrix.Set(
					IFXVector3(1.000000f, 0.000000f, 0.000000f),
					IFXVector3(0.000000f, 1.000000f, 0.000000f),
					IFXVector3(0.000000f, 0.000000f, 1.000000f),
					IFXVector3(31.295425f, -134.068436f, 19.701351f)
				  );
		result = pLightNode->SetMatrix(0, &matrix);
	}

	// Create Light resource
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXLightResource, IID_IFXLightResource,
									(void**)&pLightResource);

	// Initialize the light resource
	if (IFXSUCCESS(result))
		result = pLightResource->SetSceneGraph(pSceneGraph);

	// Set light parameters
	if (IFXSUCCESS(result))
	{	
		pLightResource->SetType(lightType);
		pLightResource->SetAttenuation(lightAttenuation);
		pLightResource->SetColor(lightColor);
		pLightResource->SetIntensity(lightIntensity);
	}

	// Get light palette from scenegraph.
	if (IFXSUCCESS(result))
		result = pSceneGraph->GetPalette(IFXSceneGraph::LIGHT, &pLightPalette);
	
	// Add an entry into the Light Resource Palette for the new light resource.
	if (IFXSUCCESS(result))
		result = pLightPalette->Add(lightResourceName, &lightResourceID);

	// Initialize the new palette entry with the light resource.
	if (IFXSUCCESS(result))
		result = pLightPalette->SetResourcePtr(lightResourceID, pLightResource);

	// Set the light to use the light resource created above.
	// This creates the scene link between the light and 
	// the light resource.
	if (IFXSUCCESS(result))
		result = pLightNode->SetLightResourceID(lightResourceID);

	// Release IFXLight Component
	IFXRELEASE(pLightPalette)
	IFXRELEASE(pLightResource)
	IFXRELEASE(pLightNode)

	/*********************************************************************************
	*
	* 	Create Model with associated mesh
	*	
	*********************************************************************************/

	IFXString modelNodeName("ModelNode1");
	U32 modelNodeID = 0;
	IFXString modelResourceName("ModelResource1");
	U32 modelResourceID = 0;
	IFXAuthorCLODResource* pAuthorClodResource = NULL;
	IFXAuthorCLODMesh* pAuthorClodMesh = NULL;
	IFXPalette* pGeneratorPalette = NULL;
	IFXModel* pModelNode = NULL;
	IFXAuthorGeomCompiler* pCompiler = NULL;

	// Create Model node component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXModel, IID_IFXModel, (void**)&pModelNode);

	// Initialize model object.
	if (IFXSUCCESS(result))
		result = pModelNode->SetSceneGraph(pSceneGraph);

	// Set the parent of the model to be the World. 
	if (IFXSUCCESS(result))
		result = pModelNode->AddParent(pWorldNode);

	// Add an entry into the Node 
	// Hierarchy Palette for the new model node.	
	if (IFXSUCCESS(result))
		result = pNodePalette->Add(modelNodeName, &modelNodeID);

	// Initialize the new palette entry with the model node.
	if (IFXSUCCESS(result))
		result = pNodePalette->SetResourcePtr(modelNodeID, pModelNode);

	// Setting the model's local tranlation matrix.
	if (IFXSUCCESS(result))
	{
		IFXMatrix4x4 matrix;
		matrix.Reset();
		matrix.Set(
					IFXVector3(1.000000f, 0.000000f, 0.000000f),
					IFXVector3(0.000000f, 1.000000f, 0.000000f),
					IFXVector3(0.000000f, 0.000000f, 1.000000f),
					IFXVector3(-3.336568f, -63.002571f, 0.000000f)
				  );
		result = pModelNode->SetMatrix(0, &matrix);
	}

	// Create Author mesh component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXAuthorMesh, IID_IFXAuthorCLODMesh,
									(void**)&pAuthorClodMesh);

	// Fill IFXAuthorMesh objects data 
	if (IFXSUCCESS(result))
		result = CreateCubeMesh(pAuthorClodMesh);

	// Create mesh compiler that will create updates for progressive CLOD mesh
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXAuthorGeomCompiler, IID_IFXAuthorGeomCompiler,
									(void**)&pCompiler);

	// Initialize mesh compiler component with SceneGraph reference. 
	if (IFXSUCCESS(result))
		result = pCompiler->SetSceneGraph(pSceneGraph);

	// Set mesh compiler parameters and compile mesh (create progressive updates)
	if (IFXSUCCESS(result))
	{
		IFXAuthorGeomCompilerParams aparams;
		// Set minimal resolution to make all mesh progressive (no base mesh)
		aparams.CompressParams.bSetMinimumResolution = TRUE;
		aparams.CompressParams.uMinimumResolution = 0;
		// Set default quality
		aparams.CompressParams.bSetDefaultQuality = TRUE;
		aparams.CompressParams.uDefaultQuality = IFX_DEFAULT_QUALITY_FACTOR;
		// Set compiler flag to use assignments above
		aparams.bCompressSettings = TRUE;
		// Progress function could be set to be called during CLOD creation process
		aparams.pProgressCallback = NULL;
		result = pCompiler->Compile(modelResourceName, pAuthorClodMesh,
									&pAuthorClodResource, FALSE, &aparams);
	}

	// Get Generator palette.
	if (IFXSUCCESS(result))
		result = pSceneGraph->GetPalette(IFXSceneGraph::GENERATOR, &pGeneratorPalette);

	// Add an entry into the Generator Resource Palette for the new AuthorClod resource.
	if (IFXSUCCESS(result))
		result = pGeneratorPalette->Add(modelResourceName, &modelResourceID);

	// Initialize the new palette entry with the Light node.
	if (IFXSUCCESS(result))
		result = pGeneratorPalette->SetResourcePtr(modelResourceID, pAuthorClodResource);

	// Record the generator palette ID of the generator in the model component.
	if (IFXSUCCESS(result))
		result = pModelNode->SetResourceIndex(modelResourceID);

	IFXRELEASE(pCompiler)
	IFXRELEASE(pNodePalette)
	IFXRELEASE(pGeneratorPalette)
	IFXRELEASE(pAuthorClodResource)
	IFXRELEASE(pAuthorClodMesh)

	/*********************************************************************************
	*
	*		Set Shader modifier on our model
	*
	*********************************************************************************/

	IFXString shaderName("ShaderLitTexture1");
	U32 shaderID = 0;
	IFXString materialName("Material1");
	U32 materialID = 0;
	IFXString textureName("Texture1");
	U32 textureID = 0;
	IFXPalette* pShaderPalette = NULL;
	IFXShaderLitTexture* pShader = NULL;
	IFXMaterialResource* pMaterialResource = NULL;  
	IFXPalette* pMaterialPalette = NULL;
	IFXShaderList* pShaderList = NULL;
	IFXShadingModifier* pShadingModifier = NULL;
	IFXModifierChain* pModChain = NULL;
	IFXTextureObject* pTextureObject = NULL;
	IFXPalette* pTexturePalette = NULL;

	// Material properties
	IFXVector4 materialAmbient(0.500000f, 0.060000f, 0.060000f, 1.00000f);
	IFXVector4 materialDiffuse(0.341176f, 0.560784f, 0.878431f, 1.00000f);
	IFXVector4 materialSpecular(0.0720000f, 0.0720000f, 0.0720000f, 1.00000f);
	IFXVector4 materialEmission(0.200000f, 0.200000f, 0.200000f, 1.00000f);
	F32 materialOpacity = 1.0f;
	F32 materialReflectivity = 0.5f;

	// Get reference to the shader palette
	if (IFXSUCCESS(result))
		result = pSceneGraph->GetPalette(IFXSceneGraph::SHADER, &pShaderPalette);

	// Get reference to the material palette
	if (IFXSUCCESS(result))
		result = pSceneGraph->GetPalette(IFXSceneGraph::MATERIAL, &pMaterialPalette);

	// Create Material resource component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXMaterialResource, IID_IFXMaterialResource,
									(void**)&pMaterialResource);

	// Initialize MaterialResource component with SceneGraph reference
	if (IFXSUCCESS(result))
		result = pMaterialResource->SetSceneGraph(pSceneGraph);

	// Set quality factor to use.
	// Also there is IFX_MAXIMUM_QUALITY_FACTOR constant equal to 1000, so
	// any value between 0 and 1000 is relative quality factor (higher - better).
	if (IFXSUCCESS(result))
		pMaterialResource->SetQualityFactorX(IFX_DEFAULT_QUALITY_FACTOR);

	// Add entry to Material Palette for the material
	if (IFXSUCCESS(result))
		result = pMaterialPalette->Add(materialName, &materialID);

	// Initialize new palette entry with material resource
	if (IFXSUCCESS(result))
		result = pMaterialPalette->SetResourcePtr(materialID, pMaterialResource);

	// Set material properties
	if (IFXSUCCESS(result))
		result = pMaterialResource->SetAmbient(materialAmbient);
	if (IFXSUCCESS(result))
		result = pMaterialResource->SetDiffuse(materialDiffuse);
	if (IFXSUCCESS(result))
		result = pMaterialResource->SetSpecular(materialSpecular);
	if (IFXSUCCESS(result))
		result = pMaterialResource->SetEmission(materialEmission);
	if (IFXSUCCESS(result))
		result = pMaterialResource->SetOpacity(materialOpacity);
	if (IFXSUCCESS(result))
		result = pMaterialResource->SetReflectivity(materialReflectivity);

	// Create IFXShaderLitTexture component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXShaderLitTexture, IID_IFXShaderLitTexture,
									(void**)&pShader);

	// Init IFXShaderLitTexture component.
	if (IFXSUCCESS(result))
		result = pShader->SetSceneGraph(pSceneGraph);

	// Set quality factor to use
	if (IFXSUCCESS(result))
		pShader->SetQualityFactorX(IFX_DEFAULT_QUALITY_FACTOR);

	// IFXShaderLitTexture component parameters set.
	if (IFXSUCCESS(result))
		result = pShader->SetMaterialID(materialID);

	// Add an entry into the Shader Palette for the new IFXShaderLitTexture component .
	if (IFXSUCCESS(result))
		result = pShaderPalette->Add(shaderName, &shaderID);

	// Initialize the new palette entry with the pShaderLitTexture component.
	if (IFXSUCCESS(result))
		result = pShaderPalette->SetResourcePtr(shaderID, pShader);

	// Init texture bitmap (32x32x24bpp)
	const U32 textureWidth = 32, textureHeight = 32, textureBPP = 3;
	const U32 bufferSize = textureWidth*textureHeight*textureBPP/sizeof(U32);
	U32 texture[bufferSize];
	U32 i, j;
	for (i = 0; i < textureHeight/2; i++)
	{
		for (j = 0; j < 12; j++)
			texture[i*24+j] = 0xFFFFFFFF;
		for (j = 12; j < 24; j += 3)
		{
			texture[i*24+j+0] = SET_ENDIAN32((U32)0xFF0000FF);
			texture[i*24+j+1] = SET_ENDIAN32((U32)0x00FF0000);
			texture[i*24+j+2] = SET_ENDIAN32((U32)0x0000FF00);
		}
	}
	for (i = 0; i < bufferSize/12; i++)
		for (j = 0; j < 6; j++)
			texture[(bufferSize/12+i)*6+j] = texture[(bufferSize/12-1-i)*6+j];
	
	// Create texture object
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXTextureObject, IID_IFXTextureObject,
									(void**)&pTextureObject);

	// Initialize the texture object.
	if (IFXSUCCESS(result))
		result = pTextureObject->SetSceneGraph(pSceneGraph);

	// Set quality factor to use
	if (IFXSUCCESS(result))
		pTextureObject->SetQualityFactorX(IFX_DEFAULT_QUALITY_FACTOR);

	// Place the bitmap into the IFXTextureObject
	if (IFXSUCCESS(result))
	{
		// Populate the fields in STextureSourceInfo with the properties of the texture.
		STextureSourceInfo sImageInfo;
		sImageInfo.m_name = textureName;
		sImageInfo.m_height = textureHeight;
		sImageInfo.m_width = textureWidth;
		sImageInfo.m_size = textureWidth*textureHeight*textureBPP;
		sImageInfo.m_imageType = IFXTextureObject::IFXTEXTUREMAP_FORMAT_RGB24;
		sImageInfo.m_pCodecCID = NULL;

		// Store the texture data and its properties in the pTextureObject.
		result = pTextureObject->SetRawImage(&sImageInfo, texture);
	}

	if (IFXSUCCESS(result))
		result = pSceneGraph->GetPalette(IFXSceneGraph::TEXTURE, &pTexturePalette);

	// Add an entry for the texture to the texture palette.
	if (IFXSUCCESS(result))
		result = pTexturePalette->Add(textureName, &textureID);

	// Point the texture palette entry to the texture object we created above.
	if (IFXSUCCESS(result))
		result = pTexturePalette->SetResourcePtr(textureID, pTextureObject);
	
	// Set enabled shader channel by flag.
	if (IFXSUCCESS(result))
		result = pShader->SetChannels(0x00000001);
	
	// Assign Texture to Shaders 0 channel. 
	if (IFXSUCCESS(result))
		result = pShader->SetTextureID(0, textureID);

	// Create ShaderList component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXShaderList, IID_IFXShaderList, (void**)&pShaderList);

	// Allocate place for 1 Shader.
	if (IFXSUCCESS(result))
		result = pShaderList->Allocate(1);

	// Set shader to shader set.
	if (IFXSUCCESS(result))
		result = pShaderList->SetShader(0, shaderID);

	// Create Shading modifier component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXShadingModifier, IID_IFXShadingModifier,
									(void**)&pShadingModifier);

	// Initialize IFXShadingModifier component with SceneGraph reference.
	if (IFXSUCCESS(result))
		result = pShadingModifier->SetSceneGraph(pSceneGraph);

	// Set Shader Set to ShadingModifier.
	if (IFXSUCCESS(result))
		result = pShadingModifier->SetElementShaderList(0, pShaderList);

	// Add it to the end of the modifier chain associated with the IFXModel.
	if (IFXSUCCESS(result))
		result = pModelNode->GetModifierChain(&pModChain);

	if (IFXSUCCESS(result))
		result = pModChain->AddModifier(*pShadingModifier);

	// Release unneeded references.
	IFXRELEASE(pMaterialResource)
	IFXRELEASE(pModelNode)
	IFXRELEASE(pShaderPalette)
	IFXRELEASE(pMaterialPalette)
	IFXRELEASE(pShader)
	IFXRELEASE(pShaderList)
	IFXRELEASE(pShadingModifier)
	IFXRELEASE(pTexturePalette)
	IFXRELEASE(pTextureObject)

	/*********************************************************************************
	*
	*		Set Animation modifier on our model
	*
	*********************************************************************************/

	IFXString motionName("BoxMotion1");
	U32 motionID = 0;
	IFXString trackName("BoxTrack1");
	U32 trackID = 0;
	IFXString mixerName("BoxMotion1");
	U32 mixerID = 0;
	IFXPalette* pMotionPalette = NULL;
	IFXMotionResource* pMotionResource = NULL;
	IFXPalette* pMixerPalette = NULL;
	IFXMixerConstruct* pMixerConstruct = NULL;
	IFXAnimationModifier* pAnimationModifier = NULL;
	IFXKeyFrame keyFrameArray[2];

	// Create Motion resource component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXMotionResource, IID_IFXMotionResource,
									(void**)&pMotionResource);

	// Initialize IFXMotionResource component with SceneGraph
	if (IFXSUCCESS(result))
		result = pMotionResource->SetSceneGraph(pSceneGraph);

	// Set quality factor to use
	if (IFXSUCCESS(result))
		pMotionResource->SetQualityFactorX(IFX_DEFAULT_QUALITY_FACTOR);

	// Get motion palette
	if (IFXSUCCESS(result))
		result = pSceneGraph->GetPalette(IFXSceneGraph::MOTION, &pMotionPalette);

	// Add an entry for the our MotionResource component to Motion palette
	if (IFXSUCCESS(result))
		result = pMotionPalette->Add(motionName, &motionID);

	// Point the motion palette entry to the MotionResource component we created above.
	if (IFXSUCCESS(result))
		result = pMotionPalette->SetResourcePtr(motionID, pMotionResource);

	// Fill KeyFrame array
	if (IFXSUCCESS(result))
	{
		IFXQuaternion vRotation1, vRotation2;
		vRotation1.MakeRotation(0, IFX_Z_AXIS);
		vRotation2.MakeRotation(IFXPI, IFX_Z_AXIS);
		IFXVector3 vScale(1.0f, 1.0f, 1.0f);
		IFXVector3 vLocation(0.0f, 0.0f, 0.0f);

		// Set time key
		keyFrameArray[0].SetTime(0.0f);
		// Set location 
		keyFrameArray[0].Location() = vLocation;
		// Set rotation
		keyFrameArray[0].Rotation() = vRotation1;
		// Set scale
		keyFrameArray[0].Scale() = vScale;

		keyFrameArray[1].SetTime(5.0f);
		keyFrameArray[1].Location() = vLocation;
		keyFrameArray[1].Rotation() = vRotation2;
		keyFrameArray[1].Scale() = vScale;
	}

	// Add a new track by name. The track index is returned.
	if (IFXSUCCESS(result))
		result = pMotionResource->AddTrack(&trackName, &trackID);

	// Remove all key frames from a track.
	if (IFXSUCCESS(result))
		result = pMotionResource->ClearTrack(trackID);

	// Set the key frame array to the motion resource
	if (IFXSUCCESS(result))
		result = pMotionResource->InsertKeyFrames(trackID, 2, keyFrameArray);

	// Create Mixer Construct component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXMixerConstruct, IID_IFXMixerConstruct,
									(void**)&pMixerConstruct);

	// Initialize IFXMixerConstruct component with SceneGraph
	if (IFXSUCCESS(result))
		result = pMixerConstruct->SetSceneGraph(pSceneGraph);

	// Set quality factor to use
	if (IFXSUCCESS(result))
		pMixerConstruct->SetQualityFactorX(IFX_DEFAULT_QUALITY_FACTOR);

	// Set motion resource to the IFXMixerConstruct component
	if (IFXSUCCESS(result))
		pMixerConstruct->SetMotionResource(pMotionResource);

	// Get Mixer Palette
	if (IFXSUCCESS(result))
		result = pSceneGraph->GetPalette(IFXSceneGraph::MIXER, &pMixerPalette);

	// Add an entry for the our IFXMixerConstruct component to Mixer palette
	if (IFXSUCCESS(result))
		result = pMixerPalette->Add(mixerName, &mixerID);

	// Point the Mixer palette entry to the IFXMixerConstruct component we created above.
	if (IFXSUCCESS(result))
		result = pMixerPalette->SetResourcePtr(mixerID, pMixerConstruct);

	// Create Animation modifier component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXAnimationModifier, IID_IFXAnimationModifier,
									(void**)&pAnimationModifier);

	// Initialize IFXAnimationModifier component with SceneGraph
	if (IFXSUCCESS(result))
		result = pAnimationModifier->SetSceneGraph(pSceneGraph);

	// Inits instance as keyframe player.  
	// Can be initialized only once. Must be initialized right after creation
	if (IFXSUCCESS(result))
		pAnimationModifier->SetAsKeyframe();

	// Put the object (the new modifier) into the end of modifier chain
	if (IFXSUCCESS(result))
		result = pModChain->AddModifier(*pAnimationModifier);

	if (IFXSUCCESS(result))
	{   
		F32 fVal = 1.0f;
		BOOL bTrue = TRUE;
		// Set Playing Flag
		pAnimationModifier->Playing() = TRUE;
		// Set TimeScale
		pAnimationModifier->TimeScale() = 1.0f;
		// Add MixerConstruct to the playing queue
		result = pAnimationModifier->Queue(mixerName, NULL, NULL, NULL,
										   &fVal, &bTrue, NULL, TRUE);
	}

	// Release unneeded interfaces
	IFXRELEASE(pMotionPalette)
	IFXRELEASE(pMotionResource)
	IFXRELEASE(pMixerPalette)
	IFXRELEASE(pMixerConstruct)
	IFXRELEASE(pAnimationModifier)

	/*********************************************************************************
	*
	*	Add CLOD modifier to get ability to change CLOD resolution at runtime
	*
	*********************************************************************************/

	IFXCLODModifier* pCLODModifier = NULL;
	
	// Create CLOD modifier component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXCLODModifier, IID_IFXCLODModifier,
									(void**)&pCLODModifier);

	// Initialize IFXAnimationModifier component with SceneGraph
	if (IFXSUCCESS(result))
		result = pCLODModifier->SetSceneGraph(pSceneGraph);

	// Set CLOD level at full resolution (default setting)
	if (IFXSUCCESS(result))
		result = pCLODModifier->SetCLODLevel(1.0f);

	// Set CLOD modifier not to depent on view conditions
	if (IFXSUCCESS(result))
		result = pCLODModifier->SetCLODScreenSpaceControllerState(FALSE);

	// Put the object (the new modifier) into the end of modifier chain
	if (IFXSUCCESS(result))
		result = pModChain->AddModifier(*pCLODModifier);

	// Release unneeded interfaces
	IFXRELEASE(pModChain)
	IFXRELEASE(pCLODModifier)

	/*********************************************************************************
	*   
	*	Save Scene
	*
	*********************************************************************************/

	IFXWriteManager* pWriteManager = NULL;
	IFXWriteBuffer* pWriteBuffer = NULL;
	IFXStdio* pStdio = NULL;

	// Create Write Manager component
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXWriteManager, IID_IFXWriteManager,
									(void**)&pWriteManager);

	// Initialize WriteManeger
	if (IFXSUCCESS(result))
		result = pWriteManager->Initialize(pCoreServices);

	// Create an IFXWriteBuffer object
	if (IFXSUCCESS(result))
		result = IFXCreateComponent(CID_IFXStdioWriteBuffer, IID_IFXWriteBuffer,
									(void**)&pWriteBuffer);

	// Get the objects's IFXStdio interface.
	if (IFXSUCCESS(result))
		result = pWriteBuffer->QueryInterface(IID_IFXStdio, (void**)&pStdio);

	// Open file to store scene
	if (IFXSUCCESS(result))
		result = pStdio->Open(L"output.u3d");

	// Mark all subnodes to store. (Recursively marks a subset of the database)
	if (IFXSUCCESS(result))
		result = pSceneGraph->Mark();

	// Write an SceneGraph out to an WriteBuffer, based on the options supplied
	// in exportOptions.
	if (IFXSUCCESS(result))
		result = pWriteManager->Write(pWriteBuffer, IFXEXPORT_EVERYTHING);

	// Close the file
	if (IFXSUCCESS(result))
		result = pStdio->Close();

	IFXRELEASE(pStdio)
	IFXRELEASE(pWriteBuffer)
	IFXRELEASE(pWriteManager)

	/*********************************************************************************
	*
	*		END OF WORK.
	*
	*********************************************************************************/

	IFXRELEASE(pWorldNode)
	IFXRELEASE(pSceneGraph)
	IFXRELEASE(pCoreServices)

	return result;
}


IFXRESULT CreateCubeMesh(IFXAuthorCLODMesh* pClodMesh)
{
	if (!pClodMesh) return IFX_E_INVALID_POINTER;
	IFXRESULT result = IFX_OK;

	// Positions
	F32 positions[] =
	{
		-20.000000, -20.000000, 0.000000,
		20.000000, -20.000000, 0.000000,
		-20.000000, 20.000000, 0.000000,
		20.000000, 20.000000, 0.000000,
		-20.000000, -20.000000, 40.000000,
		20.000000, -20.000000, 40.000000,
		-20.000000, 20.000000, 40.000000,
		20.000000, 20.000000, 40.000000
	};
	U32 positionsCount = sizeof(positions) / sizeof(F32) / 3;

	// Face positions
	U32 meshFaces[] = 
	{
		0, 2, 3,
		3, 1, 0,
		4, 5, 7,
		7, 6, 4,
		0, 1, 5,
		5, 4, 0,
		1, 3, 7,
		7, 5, 1,
		3, 2, 6,
		6, 7, 3,
		2, 0, 4,
		4, 6, 2
	};
	U32 facesCount = sizeof(meshFaces) / sizeof(U32) / 3;

	// Normals
	F32 normals[] =
	{
		0.000000, 0.000000, -1.000000,
		0.000000, 0.000000, -1.000000,
		0.000000, 0.000000, -1.000000,
		0.000000, 0.000000, -1.000000,
		0.000000, 0.000000, -1.000000,
		0.000000, 0.000000, -1.000000,

		0.000000, 0.000000, 1.000000,
		0.000000, 0.000000, 1.000000,
		0.000000, 0.000000, 1.000000,
		0.000000, 0.000000, 1.000000,
		0.000000, 0.000000, 1.000000,
		0.000000, 0.000000, 1.000000,

		0.000000, -1.000000, 0.000000,
		0.000000, -1.000000, 0.000000,
		0.000000, -1.000000, 0.000000,
		0.000000, -1.000000, 0.000000,
		0.000000, -1.000000, 0.000000,
		0.000000, -1.000000, 0.000000,

		1.000000, 0.000000, 0.000000,
		1.000000, 0.000000, 0.000000,
		1.000000, 0.000000, 0.000000,
		1.000000, 0.000000, 0.000000,
		1.000000, 0.000000, 0.000000,
		1.000000, 0.000000, 0.000000,

		0.000000, 1.000000, 0.000000,
		0.000000, 1.000000, 0.000000,
		0.000000, 1.000000, 0.000000,
		0.000000, 1.000000, 0.000000,
		0.000000, 1.000000, 0.000000,
		0.000000, 1.000000, 0.000000,

		-1.000000, 0.000000, 0.000000,
		-1.000000, 0.000000, 0.000000,
		-1.000000, 0.000000, 0.000000,
		-1.000000, 0.000000, 0.000000,
		-1.000000, 0.000000, 0.000000,
		-1.000000, 0.000000, 0.000000
	};
	U32 normalsCount = sizeof(normals) / sizeof(F32) / 3;

	// Face normals
	U32 faceNormals[] = 
	{
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11,
		12, 13, 14,
		15, 16, 17,
		18, 19, 20,
		21, 22, 23,
		24, 25, 26,
		27, 28, 29,
		30, 31, 32,
		33, 34, 35
	};
	U32 faceNormalsCount = sizeof(faceNormals) / sizeof(U32) / 3;

	// Texture coordinates
	IFXVector4	tc1(0.0f,0.0f,0.0f,0.0f),
				tc2(1.0f,0.0f,0.0f,0.0f),
				tc3(1.0f,1.0f,0.0f,0.0f),
				tc4(0.0f,1.0f,0.0f,0.0f);
	U32 texCoordsCount = 4;

	// Initialize AuthorMesh descriptor
	IFXAuthorMeshDesc desc;
	desc.NumBaseVertices = 0;
	desc.NumPositions = positionsCount;
	desc.NumNormals = normalsCount;
	desc.NumFaces = facesCount;
	desc.NumSpecularColors = 0;
	desc.NumTexCoords = texCoordsCount;
	desc.NumMaterials = 1;
	desc.NumDiffuseColors = 0;

	// Allocate memory for mesh
	if (IFXSUCCESS(result))
		result = pClodMesh->Allocate(&desc);

	// Initialize AuthorMesh
	if (IFXSUCCESS(result))
		result = pClodMesh->SetMeshDesc(&desc);

	// Set positions
	IFXVector3 vector;
	U32 i;
	for (i = 0; (i < positionsCount) && IFXSUCCESS(result); i++)
	{
		vector.X() = positions[i*3];
		vector.Y() = positions[i*3+1];
		vector.Z() = positions[i*3+2];
		result = pClodMesh->SetPosition(i, &vector);
	}

	// Set face positions
	IFXAuthorFace face;
	for (i = 0; (i < facesCount) && IFXSUCCESS(result); i++)
	{
		face.Set(meshFaces[i*3], meshFaces[i*3+1], meshFaces[i*3+2]);
		result = pClodMesh->SetPositionFace(i, &face);
	}

	// Set normals
	for (i = 0; (i < normalsCount) && IFXSUCCESS(result); i++)
	{
		vector.X() = normals[i*3];
		vector.Y() = normals[i*3+1];
		vector.Z() = normals[i*3+2];
		result = pClodMesh->SetNormal(i, &vector);
	}

	// Set face normals
	for (i = 0; (i < faceNormalsCount) && IFXSUCCESS(result); i++)
	{
		face.Set(faceNormals[i*3], faceNormals[i*3+1], faceNormals[i*3+2]);
		result = pClodMesh->SetNormalFace(i, &face);
	}

	// Set material
	if (IFXSUCCESS(result))
	{
		// Determine the material we will use.
		IFXAuthorMaterial material;
		material.m_uNumTextureLayers		= 1;
		material.m_uTexCoordDimensions[0]	= 2;
		material.m_uOriginalMaterialID		= 0;
		material.m_uDiffuseColors			= FALSE;
		material.m_uSpecularColors			= FALSE;
		material.m_uNormals					= TRUE;

		result = pClodMesh->SetMaterial(0, &material);
	}

	// Set texture coordinates
	if (IFXSUCCESS(result))
		result = pClodMesh->SetTexCoord(0, &tc1);
	if (IFXSUCCESS(result))
		result = pClodMesh->SetTexCoord(1, &tc2);
	if (IFXSUCCESS(result))
		result = pClodMesh->SetTexCoord(2, &tc3);
	if (IFXSUCCESS(result))
		result = pClodMesh->SetTexCoord(3, &tc4);

	// Set face texture coordinates
	for (i = 0; (i < facesCount) && IFXSUCCESS(result); ++i)
	{
		IFXAuthorFace texFace;
		
		if ( i % 2 == 0 )
			texFace.Set(0, 1, 2);
		else
			texFace.Set(2, 3, 0);

		result = pClodMesh->SetTexFace(0, i, &texFace);
	}



	// Set face materials
	for (i = 0; (i < facesCount) && IFXSUCCESS(result); i++)
	{
		result = pClodMesh->SetFaceMaterial(i, 0);
	}

	return result;
}
