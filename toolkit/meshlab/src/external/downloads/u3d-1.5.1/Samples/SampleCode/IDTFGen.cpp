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
  @file ConverterDriver.cpp

      This module defines driver for IDTF converter. It contains main
    function.
*/


//***************************************************************************
//  Defines
//***************************************************************************

//***************************************************************************
//  Includes
//***************************************************************************

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <wchar.h>
#include <math.h>


#include "IFXResult.h"
#include "IFXOSLoader.h"

#include "ConverterResult.h"
#include "IFXDebug.h"
#include "IFXCOM.h"

#include "ConverterOptions.h"
#include "SceneConverterLib.h"
#include "SceneUtilities.h"
#include "IFXOSUtilities.h"

#include "File.h"
#include "Tokens.h"
#include "Point.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

using namespace U3D_IDTF;

//***************************************************************************
//  Constants
//***************************************************************************


//***************************************************************************
//  Enumerations
//***************************************************************************


//***************************************************************************
//  Classes, structures and types
//***************************************************************************


//***************************************************************************
//  Global data
//***************************************************************************


//***************************************************************************
//  Local data
//***************************************************************************


//***************************************************************************
//  Local function prototypes
//***************************************************************************


//***************************************************************************
//  Public methods
//***************************************************************************


//***************************************************************************
//  Protected methods
//***************************************************************************


//***************************************************************************
//  Private methods
//***************************************************************************


//***************************************************************************
//  Global functions
//***************************************************************************

/**
  This is the driver for IDTF converter.

  @param   int argc    The number of arguments in the command line.
  @param   char *argv[]  The arguments themselves, including the command itself.

  @return  int   The last result code.
*/

int main()
{		
	IFXRESULT result = IFX_OK;

	result = IFXSetDefaultLocale();
	IFXTRACE_GENERIC(L"[Converter] IFXSetDefaultLocale %i\n", result);

	if( IFXSUCCESS(result) )
	{
		IFXDEBUG_STARTUP();
		result = IFXCOMInitialize();
	}

	{
	ConverterOptions converterOptions;
	FileOptions fileOptions;

	if( IFXSUCCESS(result) )
	{
		fileOptions.outFile		= L"test.u3d";
		fileOptions.exportOptions	= IFXExportOptions(65535);
		fileOptions.profile		= 0;
		fileOptions.scalingFactor	= 1.0f;
		fileOptions.debugLevel		= 1;

		converterOptions.positionQuality	= 1000;
		converterOptions.texCoordQuality	= 1000;
		converterOptions.normalQuality		= 1000;
		converterOptions.diffuseQuality		= 1000;
		converterOptions.specularQuality	= 1000;
		converterOptions.geoQuality		= 1000;
		converterOptions.textureQuality		= 100;
		converterOptions.animQuality		= 1000;
		converterOptions.textureLimit		= 0;
		converterOptions.removeZeroAreaFaces	= TRUE;
		converterOptions.zeroAreaFaceTolerance	= 100.0f * FLT_EPSILON;
		converterOptions.excludeNormals		= FALSE;
	}


	SceneUtilities sceneUtils;

	if( IFXSUCCESS(result) )
		result = sceneUtils.InitializeScene( fileOptions.profile, fileOptions.scalingFactor );


	SceneConverter converter( &sceneUtils, &converterOptions );

//	Simpliest box
	if( IFXSUCCESS(result) && 0 )
	{

		NodeList& Nodes = converter.m_nodeList;

		{
			ViewNode View;
			View.SetType( IDTF_VIEW );
			View.SetName( L"DefaultView" );
			View.SetResourceName( L"SceneViewResource" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
//			IFXMatrix4x4 Matrix;
//			Matrix.Reset();
//			Matrix.Rotate3x4( (75.0/180.0)*M_PI, IFX_X_AXIS );
//			Matrix.Translate3x4( IFXVector3( 0.0, 0.0, 250.0 ) );
			const float matrix[16] = 
				{
					1.000000f,    0.000000f,  0.000000f,  0.000000f,     
					0.000000f,    0.258819f,  0.965926f,  0.000000f,     
					0.000000f,   -0.965926f,  0.258819f,  0.000000f,     
					0.000000f, -241.481461f, 64.704765f,  1.000000f  
				};
			IFXMatrix4x4 Matrix = IFXMatrix4x4( matrix );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			View.SetParentList( Parents );
			ViewNodeData ViewData;
			ViewData.SetUnitType( IDTF_VIEW_UNIT_PIXEL );
			ViewData.SetClipping( VIEW_NEAR_CLIP, VIEW_FAR_CLIP );
			ViewData.SetViewPort( VIEW_PORT_WIDTH, VIEW_PORT_HEIGHT, VIEW_PORT_H_POSITION, VIEW_PORT_V_POSITION );
			ViewData.SetType( IDTF_PERSPECTIVE_VIEW );
			ViewData.SetProjection( 34.515877f );
			View.SetViewData( ViewData );
			Nodes.AddNode( &View );
		}
		
		{
			ModelNode Model;
			Model.SetType( IDTF_MODEL );
			Model.SetName( L"Box01" );
			Model.SetResourceName( L"Box01" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			const float matrix[16] = 
			{
				1.000000f, 0.000000f, 0.000000f, 0.000000f,
				0.000000f, 1.000000f, 0.000000f, 0.000000f,
				0.000000f, 0.000000f, 1.000000f, 0.000000f,
				-3.336568f, -63.002571f, 0.000000f, 1.000000f
			};
			IFXMatrix4x4 Matrix = IFXMatrix4x4( matrix );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Model.SetParentList( Parents );
			Nodes.AddNode( &Model );
		}
		
		{
			LightNode Light;
			Light.SetType( IDTF_LIGHT );
			Light.SetName( L"Omni01" );
			Light.SetResourceName( L"DefaultPointLight" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			IFXVector3 Translation;
			Translation.Set( 31.295425f, -134.068436f, 19.701351f );
			Matrix.SetTranslation( Translation );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Light.SetParentList( Parents );
			Nodes.AddNode( &Light );
		}
		
		SceneResources& Resources = converter.m_sceneResources;
		
		{
			ViewResourceList* pViewResources = static_cast< ViewResourceList* >( Resources.GetResourceList( IDTF_VIEW ) );
			pViewResources->SetType( IDTF_VIEW );
			ViewResource defaultViewResource;
			defaultViewResource.SetName( L"SceneViewResource" );
			defaultViewResource.AddRootNode( L"" );
			pViewResources->AddResource( defaultViewResource );
		}
		
		{
			LightResourceList* pLightResources = static_cast< LightResourceList* >( Resources.GetResourceList( IDTF_LIGHT ) );
			pLightResources->SetType( IDTF_LIGHT );
			LightResource lightResource;
			lightResource.SetName( L"DefaultPointLight" );
			lightResource.m_type = IDTF_POINT_LIGHT;
			lightResource.m_color.SetColor( IFXVector4( 1.0f, 1.0f, 1.0f ) );
			lightResource.m_attenuation.SetPoint( IFXVector3( 1.0f, 0.0f, 0.0f ) );
			lightResource.m_intensity = 1.0f;
			lightResource.m_spotAngle = 0.0f;
			pLightResources->AddResource( lightResource );
		}
		
		{
			ModelResourceList* pModelResources = static_cast< ModelResourceList* >( Resources.GetResourceList( IDTF_MODEL ) );
			pModelResources->SetType( IDTF_MODEL );
			MeshResource meshResource;
			meshResource.SetName( L"Box01" );
			meshResource.m_type = IDTF_MESH;
			meshResource.faceCount = 12;
			meshResource.m_modelDescription.positionCount = 8;
			meshResource.m_modelDescription.basePositionCount = 0;
			meshResource.m_modelDescription.normalCount = 36;
			meshResource.m_modelDescription.diffuseColorCount = 0;
			meshResource.m_modelDescription.specularColorCount = 0;
			meshResource.m_modelDescription.textureCoordCount = 0;
			meshResource.m_modelDescription.boneCount = 0;
			meshResource.m_modelDescription.shadingCount = 1;
			ShadingDescription shadingDescription;
			shadingDescription.m_shaderId = 0;
			shadingDescription.m_textureLayerCount = 0;
			meshResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );
			unsigned facePositionList [12][3] = 
			{
				{ 0, 2, 3 }, 
				{ 3, 1, 0 }, 
				{ 4, 5, 7 }, 
				{ 7, 6, 4 }, 
				{ 0, 1, 5 }, 
				{ 5, 4, 0 }, 
				{ 1, 3, 7 }, 
				{ 7, 5, 1 }, 
				{ 3, 2, 6 }, 
				{ 6, 7, 3 }, 
				{ 2, 0, 4 }, 
				{ 4, 6, 2 } 
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_facePositions.CreateNewElement().SetData( facePositionList[faceIndex][0], facePositionList[faceIndex][1], facePositionList[faceIndex][2] );
			unsigned faceNormalList[12][3] =
			{
				{  0,  1,  2 },
				{  3,  4,  5 },
				{  6,  7,  8 },
				{  9, 10, 11 },
				{ 12, 13, 14 },
				{ 15, 16, 17 },
				{ 18, 19, 20 },
				{ 21, 22, 23 },
				{ 24, 25, 26 },
				{ 27, 28, 29 },
				{ 30, 31, 32 },
				{ 33, 34, 35 }
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_faceNormals.CreateNewElement().SetData( faceNormalList[faceIndex][0], faceNormalList[faceIndex][1], faceNormalList[faceIndex][2] );
			unsigned faceShadingList[12] =
			{
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_faceShaders.CreateNewElement() = faceShadingList[faceIndex];
			float positionList[8][3] =
			{
				{ -20.000000, -20.000000,  0.000000 },
				{  20.000000, -20.000000,  0.000000 }, 
				{ -20.000000,  20.000000,  0.000000 },
				{  20.000000,  20.000000,  0.000000 },
				{ -20.000000, -20.000000, 40.000000 },
				{  20.000000, -20.000000, 40.000000 },
				{ -20.000000,  20.000000, 40.000000 },
				{  20.000000,  20.000000, 40.000000 }
			};
			for( int positionIndex = 0; positionIndex < meshResource.m_modelDescription.positionCount; positionIndex++ )
				meshResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( positionList[positionIndex][0], positionList[positionIndex][1], positionList[positionIndex][2] ) );
			float normalList[36][3] =
			{
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 }
			};
			for( int normalIndex = 0; normalIndex < meshResource.m_modelDescription.normalCount; normalIndex++ )
				meshResource.m_normals.CreateNewElement().SetPoint( IFXVector3 ( normalList[normalIndex][0], normalList[normalIndex][1], normalList[normalIndex][2] ) );
			
			pModelResources->AddResource( &meshResource );
		}
		
		{
			ShaderResourceList* pShaderResources = static_cast< ShaderResourceList* >( Resources.GetResourceList( IDTF_SHADER ) );
			pShaderResources->SetType( IDTF_SHADER );
			Shader shaderResource;
			shaderResource.SetName( L"Box010" );
			shaderResource.m_materialName = L"Material";
			pShaderResources->AddResource( shaderResource );
		}
		
		{
			MaterialResourceList* pMaterialResources = static_cast< MaterialResourceList* >( Resources.GetResourceList( IDTF_MATERIAL ) );
			pMaterialResources->SetType( IDTF_MATERIAL );
			Material materialResource;
			materialResource.SetName( L"Material" );
			materialResource.m_ambient.SetColor(  IFXVector4( 0.180000f, 0.060000f, 0.060000f ) );
			materialResource.m_diffuse.SetColor(  IFXVector4( 0.878431f, 0.560784f, 0.341176f ) );
			materialResource.m_specular.SetColor( IFXVector4( 0.072000f, 0.072000f, 0.072000f ) );
			materialResource.m_emissive.SetColor( IFXVector4( 0.320000f, 0.320000f, 0.320000f ) );
			materialResource.m_reflectivity = 0.1f;
			materialResource.m_opacity = 1.0f;
			pMaterialResources->AddResource( materialResource );
		}
		
		ModifierList& Modifiers = converter.m_modifierList;
		
		{
			ShadingModifier shadingModifier;
			shadingModifier.SetName( L"Box01" );
			shadingModifier.SetType( IDTF_SHADING_MODIFIER );
			shadingModifier.SetChainType( IDTF_NODE );
			shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
			ShaderList shaderList;
			shaderList.AddShaderName( L"Box010" );
			shadingModifier.AddShaderList( shaderList );
			Modifiers.AddModifier( &shadingModifier );
		}
	}

//	Simpliest box with vertex color
	if( IFXSUCCESS(result) && 0 )
	{

		NodeList& Nodes = converter.m_nodeList;

		{
			ViewNode View;
			View.SetType( IDTF_VIEW );
			View.SetName( L"DefaultView" );
			View.SetResourceName( L"SceneViewResource" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			Matrix.Rotate3x4( (75.0f/180.0f)*M_PI, IFX_X_AXIS );
			Matrix.Translate3x4( IFXVector3( 0.0, 0.0, 25.0 ) );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			View.SetParentList( Parents );
			ViewNodeData ViewData;
			ViewData.SetUnitType( IDTF_VIEW_UNIT_PIXEL );
			ViewData.SetClipping( VIEW_NEAR_CLIP, VIEW_FAR_CLIP );
			ViewData.SetViewPort( VIEW_PORT_WIDTH, VIEW_PORT_HEIGHT, VIEW_PORT_H_POSITION, VIEW_PORT_V_POSITION );
			ViewData.SetType( IDTF_PERSPECTIVE_VIEW );
			ViewData.SetProjection( 34.515877f );
			View.SetViewData( ViewData );
			Nodes.AddNode( &View );
		}
		
		{
			ModelNode Model;
			Model.SetType( IDTF_MODEL );
			Model.SetName( L"Box01" );
			Model.SetResourceName( L"Box01" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			const float matrix[16] = 
			{
				1.0, 0.0, 0.0, 0.0,
				0.0, 1.0, 0.0, 0.0,
				0.0, 0.0, 1.0, 0.0,
				0.0, 0.0, 0.0, 1.0
			};
			IFXMatrix4x4 Matrix = IFXMatrix4x4( matrix );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Model.SetParentList( Parents );
			Nodes.AddNode( &Model );
		}
		
		{
			LightNode Light;
			Light.SetType( IDTF_LIGHT );
			Light.SetName( L"Omni01" );
			Light.SetResourceName( L"DefaultPointLight" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			IFXVector3 Translation;
			Translation.Set( 31.295425f, -134.068436f, 19.701351f );
			Matrix.SetTranslation( Translation );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Light.SetParentList( Parents );
			Nodes.AddNode( &Light );
		}
		
		SceneResources& Resources = converter.m_sceneResources;
		
		{
			ViewResourceList* pViewResources = static_cast< ViewResourceList* >( Resources.GetResourceList( IDTF_VIEW ) );
			pViewResources->SetType( IDTF_VIEW );
			ViewResource defaultViewResource;
			defaultViewResource.SetName( L"SceneViewResource" );
			defaultViewResource.AddRootNode( L"" );
			pViewResources->AddResource( defaultViewResource );
		}
		
		{
			LightResourceList* pLightResources = static_cast< LightResourceList* >( Resources.GetResourceList( IDTF_LIGHT ) );
			pLightResources->SetType( IDTF_LIGHT );
			LightResource lightResource;
			lightResource.SetName( L"DefaultPointLight" );
			lightResource.m_type = IDTF_POINT_LIGHT;
			lightResource.m_color.SetColor( IFXVector4( 1.0f, 1.0f, 1.0f ) );
			lightResource.m_attenuation.SetPoint( IFXVector3( 1.0f, 0.0f, 0.0f ) );
			lightResource.m_intensity = 1.0f;
			lightResource.m_spotAngle = 0.0f;
			pLightResources->AddResource( lightResource );
		}
		
		{
			ModelResourceList* pModelResources = static_cast< ModelResourceList* >( Resources.GetResourceList( IDTF_MODEL ) );
			pModelResources->SetType( IDTF_MODEL );
			MeshResource meshResource;
			meshResource.SetName( L"Box01" );
			meshResource.m_type = IDTF_MESH;
			meshResource.faceCount = 12;
			meshResource.m_modelDescription.positionCount = 8;
			meshResource.m_modelDescription.basePositionCount = 0;
			meshResource.m_modelDescription.normalCount = 6;
			meshResource.m_modelDescription.diffuseColorCount = 2;
			meshResource.m_modelDescription.specularColorCount = 0;
			meshResource.m_modelDescription.textureCoordCount = 0;
			meshResource.m_modelDescription.boneCount = 0;
			meshResource.m_modelDescription.shadingCount = 1;
			ShadingDescription shadingDescription;
			shadingDescription.m_shaderId = 0;
			shadingDescription.m_textureLayerCount = 0;
			meshResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );
			unsigned facePositionList [12][3] = 
			{
				{ 0, 2, 3 }, 
				{ 3, 1, 0 }, 
				{ 4, 5, 7 }, 
				{ 7, 6, 4 }, 
				{ 0, 1, 5 }, 
				{ 5, 4, 0 }, 
				{ 1, 3, 7 }, 
				{ 7, 5, 1 }, 
				{ 3, 2, 6 }, 
				{ 6, 7, 3 }, 
				{ 2, 0, 4 }, 
				{ 4, 6, 2 } 
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_facePositions.CreateNewElement().SetData( facePositionList[faceIndex][0], facePositionList[faceIndex][1], facePositionList[faceIndex][2] );
			unsigned faceNormalList[12][3] =
			{
				{ 0, 0, 0 },
				{ 0, 0, 0 },
				{ 1, 1, 1 },
				{ 1, 1, 1 },
				{ 2, 2, 2 },
				{ 2, 2, 2 },
				{ 3, 3, 3 },
				{ 3, 3, 3 },
				{ 4, 4, 4 },
				{ 4, 4, 4 },
				{ 5, 5, 5 },
				{ 5, 5, 5 }
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_faceNormals.CreateNewElement().SetData( faceNormalList[faceIndex][0], faceNormalList[faceIndex][1], faceNormalList[faceIndex][2] );
			unsigned faceShadingList[12] =
			{
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_faceShaders.CreateNewElement() = faceShadingList[faceIndex];
			unsigned faceDiffuseColorList[12][3] =
			{
				{ 0, 0, 0 }, 
				{ 0, 0, 0 }, 
				{ 1, 1, 1 }, 
				{ 1, 1, 1 }, 
				{ 0, 0, 1 }, 
				{ 1, 1, 0 }, 
				{ 0, 0, 1 }, 
				{ 1, 1, 0 }, 
				{ 0, 0, 1 }, 
				{ 1, 1, 0 }, 
				{ 0, 0, 1 }, 
				{ 1, 1, 0 } 
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_faceDiffuseColors.CreateNewElement().SetData( faceDiffuseColorList[faceIndex][0], faceDiffuseColorList[faceIndex][1], faceDiffuseColorList[faceIndex][2] );
			float positionList[8][3] =
			{
				{ -1, -1, -1 },
				{  1, -1, -1 }, 
				{ -1,  1, -1 },
				{  1,  1, -1 },
				{ -1, -1,  1 },
				{  1, -1,  1 },
				{ -1,  1,  1 },
				{  1,  1,  1 }
			};
			for( int positionIndex = 0; positionIndex < meshResource.m_modelDescription.positionCount; positionIndex++ )
				meshResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( positionList[positionIndex][0], positionList[positionIndex][1], positionList[positionIndex][2] ) );
			float normalList[36][3] =
			{
				{  0,  0, -1 },
				{  0,  0,  1 },
				{  0, -1,  0 },
				{  1,  0,  0 },
				{  0,  1,  0 },
				{ -1,  0,  0 },
			};
			for( int normalIndex = 0; normalIndex < meshResource.m_modelDescription.normalCount; normalIndex++ )
				meshResource.m_normals.CreateNewElement().SetPoint( IFXVector3 ( normalList[normalIndex][0], normalList[normalIndex][1], normalList[normalIndex][2] ) );
			float diffuseColorList[2][3] =
			{
				{  1,  0, 0 },
				{  0,  0, 1 }
			};
			for( int diffuseColorIndex = 0; diffuseColorIndex < meshResource.m_modelDescription.diffuseColorCount; diffuseColorIndex++ )
				meshResource.m_diffuseColors.CreateNewElement().SetColor( IFXVector4 ( diffuseColorList[diffuseColorIndex][0], diffuseColorList[diffuseColorIndex][1], diffuseColorList[diffuseColorIndex][2] ) );
			
			pModelResources->AddResource( &meshResource );
		}
		
		{
			ShaderResourceList* pShaderResources = static_cast< ShaderResourceList* >( Resources.GetResourceList( IDTF_SHADER ) );
			pShaderResources->SetType( IDTF_SHADER );
			Shader shaderResource;
			shaderResource.SetName( L"Box010" );
			shaderResource.m_materialName = L"Material";
			shaderResource.m_useVertexColor = IDTF_TRUE;
			pShaderResources->AddResource( shaderResource );
		}
		
		{
			MaterialResourceList* pMaterialResources = static_cast< MaterialResourceList* >( Resources.GetResourceList( IDTF_MATERIAL ) );
			pMaterialResources->SetType( IDTF_MATERIAL );
			Material materialResource;
			materialResource.SetName( L"Material" );
			materialResource.m_ambient.SetColor(  IFXVector4( 0.060f, 0.060f, 0.060f ) );
			materialResource.m_diffuse.SetColor(  IFXVector4( 1.000f, 1.000f, 1.000f ) );
			materialResource.m_specular.SetColor( IFXVector4( 0.072f, 0.072f, 0.072f ) );
			materialResource.m_emissive.SetColor( IFXVector4( 0.320f, 0.320f, 0.320f ) );
			materialResource.m_reflectivity = 0.1f;
			materialResource.m_opacity = 1.0f;
			pMaterialResources->AddResource( materialResource );
		}
		
		ModifierList& Modifiers = converter.m_modifierList;
		
		{
			ShadingModifier shadingModifier;
			shadingModifier.SetName( L"Box01" );
			shadingModifier.SetType( IDTF_SHADING_MODIFIER );
			shadingModifier.SetChainType( IDTF_NODE );
			shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
			ShaderList shaderList;
			shaderList.AddShaderName( L"Box010" );
			shadingModifier.AddShaderList( shaderList );
			Modifiers.AddModifier( &shadingModifier );
		}
	}

//	Simpliest box with external texture applied
	if( IFXSUCCESS(result) && 0 )
	{

		NodeList& Nodes = converter.m_nodeList;

		{
			ViewNode View;
			View.SetType( IDTF_VIEW );
			View.SetName( L"DefaultView" );
			View.SetResourceName( L"SceneViewResource" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
//			IFXMatrix4x4 Matrix;
//			Matrix.Reset();
//			Matrix.Rotate3x4( (75.0/180.0)*M_PI, IFX_X_AXIS );
//			Matrix.Translate3x4( IFXVector3( 0.0, 0.0, 250.0 ) );
			const float matrix[16] = 
			{
				1.000000f,    0.000000f,  0.000000f,  0.000000f,     
				0.000000f,    0.258819f,  0.965926f,  0.000000f,     
				0.000000f,   -0.965926f,  0.258819f,  0.000000f,     
				0.000000f, -241.481461f, 64.704765f,  1.000000f  
			};
			IFXMatrix4x4 Matrix = IFXMatrix4x4( matrix );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			View.SetParentList( Parents );
			ViewNodeData ViewData;
			ViewData.SetUnitType( IDTF_VIEW_UNIT_PIXEL );
			ViewData.SetClipping( VIEW_NEAR_CLIP, VIEW_FAR_CLIP );
			ViewData.SetViewPort( VIEW_PORT_WIDTH, VIEW_PORT_HEIGHT, VIEW_PORT_H_POSITION, VIEW_PORT_V_POSITION );
			ViewData.SetType( IDTF_PERSPECTIVE_VIEW );
			ViewData.SetProjection( 34.515877f );
			View.SetViewData( ViewData );
			Nodes.AddNode( &View );
		}

		{
			ModelNode Model;
			Model.SetType( IDTF_MODEL );
			Model.SetName( L"Box01" );
			Model.SetResourceName( L"BoxModel" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			const float matrix[16] = 
			{
				1.000000f, 0.000000f, 0.000000f, 0.000000f,
				0.000000f, 1.000000f, 0.000000f, 0.000000f,
				0.000000f, 0.000000f, 1.000000f, 0.000000f,
				-3.336568f, -63.002571f, 0.000000f, 1.000000f
			};
			IFXMatrix4x4 Matrix = IFXMatrix4x4( matrix );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Model.SetParentList( Parents );
			Nodes.AddNode( &Model );
		}

		{
			LightNode Light;
			Light.SetType( IDTF_LIGHT );
			Light.SetName( L"Omni01" );
			Light.SetResourceName( L"DefaultPointLight" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			IFXVector3 Translation;
			Translation.Set( 31.295425f, -134.068436f, 19.701351f );
			Matrix.SetTranslation( Translation );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Light.SetParentList( Parents );
			Nodes.AddNode( &Light );
		}

		SceneResources& Resources = converter.m_sceneResources;

		{
			ViewResourceList* pViewResources = static_cast< ViewResourceList* >( Resources.GetResourceList( IDTF_VIEW ) );
			pViewResources->SetType( IDTF_VIEW );
			ViewResource defaultViewResource;
			defaultViewResource.SetName( L"SceneViewResource" );
			defaultViewResource.AddRootNode( L"" );
			pViewResources->AddResource( defaultViewResource );
		}

		{
			LightResourceList* pLightResources = static_cast< LightResourceList* >( Resources.GetResourceList( IDTF_LIGHT ) );
			pLightResources->SetType( IDTF_LIGHT );
			LightResource lightResource;
			lightResource.SetName( L"DefaultPointLight" );
			lightResource.m_type = IDTF_POINT_LIGHT;
			lightResource.m_color.SetColor( IFXVector4( 1.0f, 1.0f, 1.0f ) );
			lightResource.m_attenuation.SetPoint( IFXVector3( 1.0f, 0.0f, 0.0f ) );
			lightResource.m_intensity = 1.0f;
			lightResource.m_spotAngle = 0.0f;
			pLightResources->AddResource( lightResource );
		}

		{
			ModelResourceList* pModelResources = static_cast< ModelResourceList* >( Resources.GetResourceList( IDTF_MODEL ) );
			pModelResources->SetType( IDTF_MODEL );
			MeshResource meshResource;
			meshResource.SetName( L"BoxModel" );
			meshResource.m_type = IDTF_MESH;
			meshResource.faceCount = 12;
			meshResource.m_modelDescription.positionCount = 8;
			meshResource.m_modelDescription.basePositionCount = 0;
			meshResource.m_modelDescription.normalCount = 36;
			meshResource.m_modelDescription.diffuseColorCount = 0;
			meshResource.m_modelDescription.specularColorCount = 0;
			meshResource.m_modelDescription.textureCoordCount = 4;
			meshResource.m_modelDescription.boneCount = 0;
			meshResource.m_modelDescription.shadingCount = 1;
			ShadingDescription shadingDescription;
			shadingDescription.m_shaderId = 0;
			shadingDescription.m_textureLayerCount = 1;
			shadingDescription.AddTextureCoordDimension( 2 );
			meshResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );
			unsigned facePositionList [12][3] = 
			{
				{ 0, 2, 3 }, 
				{ 3, 1, 0 }, 
				{ 4, 5, 7 }, 
				{ 7, 6, 4 }, 
				{ 0, 1, 5 }, 
				{ 5, 4, 0 }, 
				{ 1, 3, 7 }, 
				{ 7, 5, 1 }, 
				{ 3, 2, 6 }, 
				{ 6, 7, 3 }, 
				{ 2, 0, 4 }, 
				{ 4, 6, 2 } 
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_facePositions.CreateNewElement().SetData( facePositionList[faceIndex][0], facePositionList[faceIndex][1], facePositionList[faceIndex][2] );
			unsigned faceNormalList[12][3] =
			{
				{  0,  1,  2 },
				{  3,  4,  5 },
				{  6,  7,  8 },
				{  9, 10, 11 },
				{ 12, 13, 14 },
				{ 15, 16, 17 },
				{ 18, 19, 20 },
				{ 21, 22, 23 },
				{ 24, 25, 26 },
				{ 27, 28, 29 },
				{ 30, 31, 32 },
				{ 33, 34, 35 }
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_faceNormals.CreateNewElement().SetData( faceNormalList[faceIndex][0], faceNormalList[faceIndex][1], faceNormalList[faceIndex][2] );
			unsigned faceShadingList[12] =
			{
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_faceShaders.CreateNewElement() = faceShadingList[faceIndex];
			int faceTextureCoordList[12][3] =
			{
				{ 0, 1, 2 },
				{ 2, 3, 0 }, 
				{ 0, 1, 2 },
				{ 2, 3, 0 },
				{ 0, 1, 2 },
				{ 2, 3, 0 },
				{ 0, 1, 2 },
				{ 2, 3, 0 },
				{ 0, 1, 2 }, 
				{ 2, 3, 0 },
				{ 0, 1, 2 },
				{ 2, 3, 0 }
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_faceTextureCoords.CreateNewElement().m_texCoords.CreateNewElement().SetData( faceTextureCoordList[faceIndex][0], faceTextureCoordList[faceIndex][1], faceTextureCoordList[faceIndex][2] );
			float positionList[8][3] =
			{
				{ -20.000000, -20.000000,  0.000000 },
				{  20.000000, -20.000000,  0.000000 }, 
				{ -20.000000,  20.000000,  0.000000 },
				{  20.000000,  20.000000,  0.000000 },
				{ -20.000000, -20.000000, 40.000000 },
				{  20.000000, -20.000000, 40.000000 },
				{ -20.000000,  20.000000, 40.000000 },
				{  20.000000,  20.000000, 40.000000 }
			};
			for( int positionIndex = 0; positionIndex < meshResource.m_modelDescription.positionCount; positionIndex++ )
				meshResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( positionList[positionIndex][0], positionList[positionIndex][1], positionList[positionIndex][2] ) );
			float normalList[36][3] =
			{
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000, -1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000,  0.000000,  1.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  0.000000, -1.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  1.000000,  0.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{  0.000000,  1.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 },
				{ -1.000000,  0.000000,  0.000000 }
			};
			for( int normalIndex = 0; normalIndex < meshResource.m_modelDescription.normalCount; normalIndex++ )
				meshResource.m_normals.CreateNewElement().SetPoint( IFXVector3 ( normalList[normalIndex][0], normalList[normalIndex][1], normalList[normalIndex][2] ) );
			float textureCoordList[4][4] =
			{
				{ 0.000000, 0.000000, 0.000000, 0.000000 },
				{ 1.000000, 0.000000, 0.000000, 0.000000 }, 
				{ 1.000000, 1.000000, 0.000000, 0.000000 },
				{ 0.000000, 1.000000, 0.000000, 0.000000 }
			};
			for( int textureCoordIndex = 0; textureCoordIndex < meshResource.m_modelDescription.textureCoordCount; textureCoordIndex++ )
				meshResource.m_textureCoords.CreateNewElement().Set( textureCoordList[textureCoordIndex][0], textureCoordList[textureCoordIndex][1], textureCoordList[textureCoordIndex][2], textureCoordList[textureCoordIndex][3] );

			pModelResources->AddResource( &meshResource );
		}

		{
			ShaderResourceList* pShaderResources = static_cast< ShaderResourceList* >( Resources.GetResourceList( IDTF_SHADER ) );
			pShaderResources->SetType( IDTF_SHADER );
			Shader shaderResource;
			shaderResource.SetName( L"ModelShader1" );
			shaderResource.m_materialName = L"ModelMaterial";
			TextureLayer textureLayer;
			textureLayer.m_channel = 0;
			textureLayer.m_intensity = 1.0f;
			textureLayer.m_blendFunction = L"MULTIPLY";
			textureLayer.m_blendSource = L"CONSTANT";
			textureLayer.m_blendConstant = 0.5;
			textureLayer.m_textureName = L"lines";
			shaderResource.AddTextureLayer( textureLayer );
			pShaderResources->AddResource( shaderResource );
		}

		{
			MaterialResourceList* pMaterialResources = static_cast< MaterialResourceList* >( Resources.GetResourceList( IDTF_MATERIAL ) );
			pMaterialResources->SetType( IDTF_MATERIAL );
			Material materialResource;
			materialResource.SetName( L"ModelMaterial" );
			materialResource.m_ambient.SetColor(  IFXVector4( 0.180000f, 0.060000f, 0.060000f ) );
			materialResource.m_diffuse.SetColor(  IFXVector4( 0.878431f, 0.560784f, 0.341176f ) );
			materialResource.m_specular.SetColor( IFXVector4( 0.072000f, 0.072000f, 0.072000f ) );
			materialResource.m_emissive.SetColor( IFXVector4( 0.320000f, 0.320000f, 0.320000f ) );
			materialResource.m_reflectivity = 0.1f;
			materialResource.m_opacity = 1.0f;
			pMaterialResources->AddResource( materialResource );
		}

		{
			TextureResourceList* pTextureResources = static_cast< TextureResourceList* >( Resources.GetResourceList( IDTF_TEXTURE ) );
			pTextureResources->SetType( IDTF_TEXTURE );
			Texture textureResource;
			textureResource.SetName( L"lines" );
			ImageFormat imageFormat;

			imageFormat.m_blue = IDTF_TRUE;
			imageFormat.m_green = IDTF_TRUE;
			imageFormat.m_red = IDTF_TRUE;

			textureResource.AddImageFormat( imageFormat );
			textureResource.SetExternal( FALSE );
			textureResource.SetPath( L"lines.tga" );

			pTextureResources->AddResource( textureResource );
		}

		ModifierList& Modifiers = converter.m_modifierList;

		{
			ShadingModifier shadingModifier;
			shadingModifier.SetName( L"BoxModel" );
			shadingModifier.SetType( IDTF_SHADING_MODIFIER );
			shadingModifier.SetChainType( IDTF_MODEL );
			shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
			ShaderList shaderList;
			shaderList.AddShaderName( L"ModelShader1" );
			shadingModifier.AddShaderList( shaderList );
			Modifiers.AddModifier( &shadingModifier );
		}

	}

//	Two overlapping squares with transaparent textures (volume rendering test)
	if( IFXSUCCESS(result) && 0 )
	{

		NodeList& Nodes = converter.m_nodeList;

		{
			ViewNode View;
			View.SetType( IDTF_VIEW );
			View.SetName( L"DefaultView" );
			View.SetResourceName( L"SceneViewResource" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
//			Matrix.Rotate3x4( (75.0/180.0)*M_PI, IFX_X_AXIS );
			Matrix.Translate3x4( IFXVector3( 0.0, 0.0, 10.0 ) );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			View.SetParentList( Parents );
			ViewNodeData ViewData;
			ViewData.SetUnitType( IDTF_VIEW_UNIT_PIXEL );
			ViewData.SetClipping( VIEW_NEAR_CLIP, VIEW_FAR_CLIP );
			ViewData.SetViewPort( VIEW_PORT_WIDTH, VIEW_PORT_HEIGHT, VIEW_PORT_H_POSITION, VIEW_PORT_V_POSITION );
			ViewData.SetType( IDTF_PERSPECTIVE_VIEW );
			ViewData.SetProjection( 34.515877f );
			View.SetViewData( ViewData );
			Nodes.AddNode( &View );
		}

		{
			ModelNode Model;
			Model.SetType( IDTF_MODEL );
			Model.SetName( L"Box01" );
			Model.SetResourceName( L"BoxModel" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Model.SetParentList( Parents );
			Model.SetVisibility( L"BOTH" );
			Nodes.AddNode( &Model );
		}

		{
			ModelNode Model;
			Model.SetType( IDTF_MODEL );
			Model.SetName( L"Box02" );
			Model.SetResourceName( L"BoxModel" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			Matrix.Translate3x4( IFXVector3( 0.0, 0.0, 2.0 ) );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Model.SetParentList( Parents );
			Model.SetVisibility( L"BOTH" );
			Nodes.AddNode( &Model );
		}

		{
			LightNode Light;
			Light.SetType( IDTF_LIGHT );
			Light.SetName( L"Omni01" );
			Light.SetResourceName( L"DefaultPointLight" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			IFXVector3 Translation;
			Translation.Set( 31.295425f, -134.068436f, 19.701351f );
			Matrix.SetTranslation( Translation );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Light.SetParentList( Parents );
			Nodes.AddNode( &Light );
		}

		SceneResources& Resources = converter.m_sceneResources;

		{
			ViewResourceList* pViewResources = static_cast< ViewResourceList* >( Resources.GetResourceList( IDTF_VIEW ) );
			pViewResources->SetType( IDTF_VIEW );
			ViewResource defaultViewResource;
			defaultViewResource.SetName( L"SceneViewResource" );
			defaultViewResource.AddRootNode( L"" );
			pViewResources->AddResource( defaultViewResource );
		}

		{
			LightResourceList* pLightResources = static_cast< LightResourceList* >( Resources.GetResourceList( IDTF_LIGHT ) );
			pLightResources->SetType( IDTF_LIGHT );
			LightResource lightResource;
			lightResource.SetName( L"DefaultPointLight" );
			lightResource.m_type = IDTF_POINT_LIGHT;
			lightResource.m_color.SetColor( IFXVector4( 1.0f, 1.0f, 1.0f ) );
			lightResource.m_attenuation.SetPoint( IFXVector3( 1.0f, 0.0f, 0.0f ) );
			lightResource.m_intensity = 1.0f;
			lightResource.m_spotAngle = 0.0f;
			pLightResources->AddResource( lightResource );
		}

		{
			ModelResourceList* pModelResources = static_cast< ModelResourceList* >( Resources.GetResourceList( IDTF_MODEL ) );
			pModelResources->SetType( IDTF_MODEL );
			MeshResource meshResource;
			meshResource.SetName( L"BoxModel" );
			meshResource.m_type = IDTF_MESH;
			meshResource.faceCount = 2;
			meshResource.m_modelDescription.positionCount = 4;
			meshResource.m_modelDescription.basePositionCount = 0;
			meshResource.m_modelDescription.normalCount = 0;
			meshResource.m_modelDescription.diffuseColorCount = 0;
			meshResource.m_modelDescription.specularColorCount = 0;
			meshResource.m_modelDescription.textureCoordCount = 4;
			meshResource.m_modelDescription.boneCount = 0;
			meshResource.m_modelDescription.shadingCount = 1;
			ShadingDescription shadingDescription;
			shadingDescription.m_shaderId = 0;
			shadingDescription.m_textureLayerCount = 1;
			shadingDescription.AddTextureCoordDimension( 2 );
			meshResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );
			unsigned facePositionList [2][3] = 
			{
				{ 0, 1, 2 },
				{ 2, 3, 0 }
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_facePositions.CreateNewElement().SetData( facePositionList[faceIndex][0], facePositionList[faceIndex][1], facePositionList[faceIndex][2] );
			unsigned faceShadingList[2] =
			{
				0,
				0
			};
			for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
				meshResource.m_faceShaders.CreateNewElement() = faceShadingList[faceIndex];
			int faceTextureCoordList[2][3] =
			{
				{ 0, 1, 2 },
				{ 2, 3, 0 }
			};
			if ( meshResource.m_modelDescription.textureCoordCount != 0 )
				for( int faceIndex = 0; faceIndex < meshResource.faceCount; faceIndex++ )
					meshResource.m_faceTextureCoords.CreateNewElement().m_texCoords.CreateNewElement().SetData( faceTextureCoordList[faceIndex][0], faceTextureCoordList[faceIndex][1], faceTextureCoordList[faceIndex][2] );
			float positionList[4][3] =
			{
				{ -1, -1, 0 },
				{  1, -1, 0 },
				{  1,  1, 0 },
				{ -1,  1, 0 }
			};
			for( int positionIndex = 0; positionIndex < meshResource.m_modelDescription.positionCount; positionIndex++ )
				meshResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( positionList[positionIndex][0], positionList[positionIndex][1], positionList[positionIndex][2] ) );
			float textureCoordList[4][4] =
			{
				{ 0+0.5*(1/64), 0+0.5*(1/64), 0, 0 },
				{ 1-0.5*(1/64), 0+0.5*(1/64), 0, 0 }, 
				{ 1-0.5*(1/64), 1-0.5*(1/64), 0, 0 },
				{ 0+0.5*(1/64), 1-0.5*(1/64), 0, 0 }
			};
			for( int textureCoordIndex = 0; textureCoordIndex < meshResource.m_modelDescription.textureCoordCount; textureCoordIndex++ )
				meshResource.m_textureCoords.CreateNewElement().Set( textureCoordList[textureCoordIndex][0], textureCoordList[textureCoordIndex][1], textureCoordList[textureCoordIndex][2], textureCoordList[textureCoordIndex][3] );

			pModelResources->AddResource( &meshResource );
		}

		{
			ShaderResourceList* pShaderResources = static_cast< ShaderResourceList* >( Resources.GetResourceList( IDTF_SHADER ) );
			pShaderResources->SetType( IDTF_SHADER );
			Shader shaderResource;
			shaderResource.SetName( L"ModelShader1" );
			shaderResource.m_materialName = L"ModelMaterial1";
			TextureLayer textureLayer;
			textureLayer.m_channel = 0;
			textureLayer.m_intensity = 1.0f;
			textureLayer.m_blendFunction = L"REPLACE";
			textureLayer.m_blendSource = L"ALPHA";
			textureLayer.m_blendConstant = 0.5;
			textureLayer.m_alphaEnabled = IDTF_TRUE;
			textureLayer.m_repeat = L"NONE";
			textureLayer.m_textureName = L"red";
			shaderResource.AddTextureLayer( textureLayer );
			pShaderResources->AddResource( shaderResource );
		}

		{
			ShaderResourceList* pShaderResources = static_cast< ShaderResourceList* >( Resources.GetResourceList( IDTF_SHADER ) );
			//					pShaderResources->SetType( IDTF_SHADER );
			Shader shaderResource;
			shaderResource.SetName( L"ModelShader2" );
			shaderResource.m_materialName = L"ModelMaterial2";
			TextureLayer textureLayer;
			textureLayer.m_channel = 0;
			textureLayer.m_intensity = 1.0f;
			textureLayer.m_blendFunction = L"REPLACE";
			textureLayer.m_blendSource = L"ALPHA";
			textureLayer.m_blendConstant = 0.5;
			textureLayer.m_alphaEnabled = IDTF_TRUE;
			textureLayer.m_repeat = L"NONE";
			textureLayer.m_textureName = L"green";
			shaderResource.AddTextureLayer( textureLayer );
			pShaderResources->AddResource( shaderResource );
		}

		{
			MaterialResourceList* pMaterialResources = static_cast< MaterialResourceList* >( Resources.GetResourceList( IDTF_MATERIAL ) );
			pMaterialResources->SetType( IDTF_MATERIAL );
			Material materialResource;
			materialResource.SetName( L"ModelMaterial1" );
			materialResource.m_ambient.SetColor(  IFXVector4( 0.0, 0.0, 0.0 ) );
			materialResource.m_diffuse.SetColor(  IFXVector4( 0.0, 0.0, 0.0 ) );
			materialResource.m_specular.SetColor( IFXVector4( 0.0, 0.0, 0.0 ) );
			materialResource.m_emissive.SetColor( IFXVector4( 1.0, 1.0, 1.0 ) );
			materialResource.m_reflectivity = 0.0;
			materialResource.m_opacity = 1.0;
			pMaterialResources->AddResource( materialResource );
		}

		{
			MaterialResourceList* pMaterialResources = static_cast< MaterialResourceList* >( Resources.GetResourceList( IDTF_MATERIAL ) );
			pMaterialResources->SetType( IDTF_MATERIAL );
			Material materialResource;
			materialResource.SetName( L"ModelMaterial2" );
			materialResource.m_ambient.SetColor(  IFXVector4( 0.0, 0.0, 0.0 ) );
			materialResource.m_diffuse.SetColor(  IFXVector4( 0.0, 0.0, 0.0 ) );
			materialResource.m_specular.SetColor( IFXVector4( 0.0, 0.0, 0.0 ) );
			materialResource.m_emissive.SetColor( IFXVector4( 1.0, 1.0, 1.0 ) );
			materialResource.m_reflectivity = 0.0;
			materialResource.m_opacity = 1.0;
			pMaterialResources->AddResource( materialResource );
		}

		{
			TextureResourceList* pTextureResources = static_cast< TextureResourceList* >( Resources.GetResourceList( IDTF_TEXTURE ) );
			pTextureResources->SetType( IDTF_TEXTURE );
			Texture textureResource;
			textureResource.SetName( L"red" );
			ImageFormat imageFormat;
			imageFormat.m_compressionType = IDTF_IMAGE_COMPRESSION_TYPE_PNG;
			imageFormat.m_alpha = IDTF_TRUE;
			imageFormat.m_blue = IDTF_TRUE;
			imageFormat.m_green = IDTF_TRUE;
			imageFormat.m_red = IDTF_TRUE;

			textureResource.AddImageFormat( imageFormat );
			textureResource.SetExternal( FALSE );
			textureResource.SetPath( L"red.tga" );
			textureResource.SetImageType( IDTF_IMAGE_TYPE_RGBA );
			/*
			const U8 testTexture2D[ 4 * 4 * 4 ] =
			{
				255, 0, 0, 255,		255, 0, 0, 200,		0, 0, 255, 100,		0, 0, 255,   0,
				255, 0, 0, 200,		255, 0, 0, 200,		0, 0, 255, 100,		0, 0, 255,   0,
				0, 0, 255, 100,		0, 0, 255, 100,		0, 0, 255, 100,		0, 0, 255,   0,
				0, 0, 255,   0,		0, 0, 255,   0,		0, 0, 255,   0,		0, 0, 255,   0
			};
			const U8 testTexture1D[ 4 * 1 * 4 ] =
			{
				255, 0, 0, 255,		255, 0, 0, 200,		0, 0, 255, 100,		0, 0, 255,   0,
			};
			*/
			U8 testTextureAuto[ 64 * 64 * 4 ];
			for( int x = 0; x < 64; x++)
				for( int y = 0; y < 64; y++)
				{
					testTextureAuto[ (x + y*64)*4 + 0 ] = static_cast<U8>(255*(1-fabs(0.5-x/63.0)));
					testTextureAuto[ (x + y*64)*4 + 1 ] = static_cast<U8>(255*fabs(0.5-x/63.0));
					testTextureAuto[ (x + y*64)*4 + 2 ] = static_cast<U8>(255*fabs(0.5-x/63.0));
					testTextureAuto[ (x + y*64)*4 + 3 ] = static_cast<U8>(255*(1-fabs(0.5-x/63.0)));
				}
			textureResource.m_textureImage.Initialize( 64, 64, 4 );
			textureResource.m_textureImage.SetData( testTextureAuto );

			pTextureResources->AddResource( textureResource );
		}

		{
			TextureResourceList* pTextureResources = static_cast< TextureResourceList* >( Resources.GetResourceList( IDTF_TEXTURE ) );
			//					pTextureResources->SetType( IDTF_TEXTURE );
			Texture textureResource;
			textureResource.SetName( L"green" );
			ImageFormat imageFormat;
			imageFormat.m_compressionType = IDTF_IMAGE_COMPRESSION_TYPE_PNG;
			imageFormat.m_alpha = IDTF_TRUE;
			imageFormat.m_blue = IDTF_TRUE;
			imageFormat.m_green = IDTF_TRUE;
			imageFormat.m_red = IDTF_TRUE;

			textureResource.AddImageFormat( imageFormat );
			textureResource.SetExternal( FALSE );
			textureResource.SetPath( L"green.tga" );
			textureResource.SetImageType( IDTF_IMAGE_TYPE_RGBA );
			U8 testTextureAuto[ 64 * 64 * 4 ];
			for( int x = 0; x < 64; x++)
				for( int y = 0; y < 64; y++)
				{
					testTextureAuto[ (x + y*64)*4 + 0 ] = static_cast<U8>(255*fabs(0.5-y/63.0));
					testTextureAuto[ (x + y*64)*4 + 1 ] = static_cast<U8>(255*(1-fabs(0.5-y/63.0)));
					testTextureAuto[ (x + y*64)*4 + 2 ] = static_cast<U8>(255*fabs(0.5-y/63.0));
					testTextureAuto[ (x + y*64)*4 + 3 ] = static_cast<U8>(255*(1-fabs(0.5-y/63.0)));
				}
			textureResource.m_textureImage.Initialize( 64, 64, 4 );
			textureResource.m_textureImage.SetData( testTextureAuto );

			pTextureResources->AddResource( textureResource );
		}

		ModifierList& Modifiers = converter.m_modifierList;

		{
			ShadingModifier shadingModifier;
			shadingModifier.SetName( L"Box01" );
			shadingModifier.SetType( IDTF_SHADING_MODIFIER );
			shadingModifier.SetChainType( IDTF_NODE );
			shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
			ShaderList shaderList;
			shaderList.AddShaderName( L"ModelShader1" );
			shadingModifier.AddShaderList( shaderList );
			Modifiers.AddModifier( &shadingModifier );
		}

		{
			ShadingModifier shadingModifier;
			shadingModifier.SetName( L"Box02" );
			shadingModifier.SetType( IDTF_SHADING_MODIFIER );
			shadingModifier.SetChainType( IDTF_NODE );
			shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
			ShaderList shaderList;
			shaderList.AddShaderName( L"ModelShader2" );
			shadingModifier.AddShaderList( shaderList );
			Modifiers.AddModifier( &shadingModifier );
		}

	}

// LineSet
	if( IFXSUCCESS(result) && 0)
	{

		NodeList& Nodes = converter.m_nodeList;

		{
			ViewNode View;
			View.SetType( IDTF_VIEW );
			View.SetName( L"DefaultView" );
			View.SetResourceName( L"SceneViewResource" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			//					Matrix.Rotate3x4( (75.0/180.0)*M_PI, IFX_X_AXIS );
			Matrix.Translate3x4( IFXVector3( 0.0, 0.0, 10.0 ) );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			View.SetParentList( Parents );
			ViewNodeData ViewData;
			ViewData.SetUnitType( IDTF_VIEW_UNIT_PIXEL );
			ViewData.SetClipping( VIEW_NEAR_CLIP, VIEW_FAR_CLIP );
			ViewData.SetViewPort( VIEW_PORT_WIDTH, VIEW_PORT_HEIGHT, VIEW_PORT_H_POSITION, VIEW_PORT_V_POSITION );
			ViewData.SetType( IDTF_PERSPECTIVE_VIEW );
			ViewData.SetProjection( 34.515877f );
			View.SetViewData( ViewData );
			Nodes.AddNode( &View );
		}

		{
			ModelNode Model;
			Model.SetType( IDTF_MODEL );
			Model.SetName( L"LineSet" );
			Model.SetResourceName( L"LineSetModel" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Model.SetParentList( Parents );
			Nodes.AddNode( &Model );
		}


		{
			LightNode Light;
			Light.SetType( IDTF_LIGHT );
			Light.SetName( L"Omni01" );
			Light.SetResourceName( L"DefaultPointLight" );
			ParentList Parents;
			ParentData Parent;
			Parent.SetParentName( L"<NULL>" );
			IFXMatrix4x4 Matrix;
			Matrix.Reset();
			IFXVector3 Translation;
			Translation.Set( 31.295425f, -134.068436f, 19.701351f );
			Matrix.SetTranslation( Translation );
			Parent.SetParentTM( Matrix );
			Parents.AddParentData( Parent );
			Light.SetParentList( Parents );
			Nodes.AddNode( &Light );
		}

		SceneResources& Resources = converter.m_sceneResources;

		{
			ViewResourceList* pViewResources = static_cast< ViewResourceList* >( Resources.GetResourceList( IDTF_VIEW ) );
			pViewResources->SetType( IDTF_VIEW );
			ViewResource defaultViewResource;
			defaultViewResource.SetName( L"SceneViewResource" );
			defaultViewResource.AddRootNode( L"" );
			pViewResources->AddResource( defaultViewResource );
		}

		{
			LightResourceList* pLightResources = static_cast< LightResourceList* >( Resources.GetResourceList( IDTF_LIGHT ) );
			pLightResources->SetType( IDTF_LIGHT );
			LightResource lightResource;
			lightResource.SetName( L"DefaultPointLight" );
			lightResource.m_type = IDTF_POINT_LIGHT;
			lightResource.m_color.SetColor( IFXVector4( 1.0f, 1.0f, 1.0f ) );
			lightResource.m_attenuation.SetPoint( IFXVector3( 1.0f, 0.0f, 0.0f ) );
			lightResource.m_intensity = 1.0f;
			lightResource.m_spotAngle = 0.0f;
			pLightResources->AddResource( lightResource );
		}

		{
			ModelResourceList* pModelResources = static_cast< ModelResourceList* >( Resources.GetResourceList( IDTF_MODEL ) );
			pModelResources->SetType( IDTF_MODEL );
			LineSetResource lineSetResource;
			lineSetResource.SetName( L"LineSetModel" );
			lineSetResource.m_type = IDTF_LINE_SET;
			lineSetResource.lineCount = 1;
			lineSetResource.m_modelDescription.positionCount = 2;
			lineSetResource.m_modelDescription.basePositionCount = 0;
			lineSetResource.m_modelDescription.normalCount = 0;
			lineSetResource.m_modelDescription.diffuseColorCount = 0;
			lineSetResource.m_modelDescription.specularColorCount = 0;
			lineSetResource.m_modelDescription.textureCoordCount = 2;
			lineSetResource.m_modelDescription.boneCount = 0;
			lineSetResource.m_modelDescription.shadingCount = 1;
			ShadingDescription shadingDescription;
			shadingDescription.m_shaderId = 0;
			shadingDescription.m_textureLayerCount = 1;
			shadingDescription.AddTextureCoordDimension( 1 );
			lineSetResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );
			I32 linePositionList [1][2] = 
			{
				{ 0, 1 } 
			};
			for( int lineIndex = 0; lineIndex < lineSetResource.lineCount; lineIndex++ )
				lineSetResource.m_linePositions.CreateNewElement().SetData( linePositionList[lineIndex][0], linePositionList[lineIndex][1] );
			unsigned lineShadingList[1] =
			{
				0
			};
			for( int lineIndex = 0; lineIndex < lineSetResource.lineCount; lineIndex++ )
				lineSetResource.m_lineShaders.CreateNewElement() = lineShadingList[lineIndex];
			int lineTextureCoordList[12][3] =
			{
				{ 0, 1 }
			};
			if ( lineSetResource.m_modelDescription.textureCoordCount != 0 )
				for( int lineIndex = 0; lineIndex < lineSetResource.lineCount; lineIndex++ )
					lineSetResource.m_lineTextureCoords.CreateNewElement().m_texCoords.CreateNewElement().SetData( lineTextureCoordList[lineIndex][0], lineTextureCoordList[lineIndex][1] );
			float positionList[2][3] =
			{
				{ 0, 0, 0 },
				{ 1, 0, 0 }
			};
			for( int positionIndex = 0; positionIndex < lineSetResource.m_modelDescription.positionCount; positionIndex++ )
				lineSetResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( positionList[positionIndex][0], positionList[positionIndex][1], positionList[positionIndex][2] ) );
			float textureCoordList[2][4] =
			{
				{ 0+0.5*(1/2.), 0, 0, 0 },
				{ 1-0.5*(1/2.), 0, 0, 0 }
			};
			for( int textureCoordIndex = 0; textureCoordIndex < lineSetResource.m_modelDescription.textureCoordCount; textureCoordIndex++ )
				lineSetResource.m_textureCoords.CreateNewElement().Set( textureCoordList[textureCoordIndex][0], textureCoordList[textureCoordIndex][1], textureCoordList[textureCoordIndex][2], textureCoordList[textureCoordIndex][3] );

			pModelResources->AddResource( &lineSetResource );
		}

		{
			ShaderResourceList* pShaderResources = static_cast< ShaderResourceList* >( Resources.GetResourceList( IDTF_SHADER ) );
			pShaderResources->SetType( IDTF_SHADER );
			Shader shaderResource;
			shaderResource.SetName( L"ModelShader1" );
			shaderResource.m_materialName = L"ModelMaterial";
			TextureLayer textureLayer;
			textureLayer.m_channel = 0;
			textureLayer.m_intensity = 1.0f;
			textureLayer.m_blendFunction = L"REPLACE";
			textureLayer.m_blendSource = L"ALPHA";
			textureLayer.m_blendConstant = 0.5;
			textureLayer.m_alphaEnabled = IDTF_TRUE;
			textureLayer.m_repeat = L"NONE";
			textureLayer.m_textureName = L"lines";
			shaderResource.AddTextureLayer( textureLayer );
			pShaderResources->AddResource( shaderResource );
		}

		{
			MaterialResourceList* pMaterialResources = static_cast< MaterialResourceList* >( Resources.GetResourceList( IDTF_MATERIAL ) );
			pMaterialResources->SetType( IDTF_MATERIAL );
			Material materialResource;
			materialResource.SetName( L"ModelMaterial" );
			materialResource.m_ambient.SetColor(  IFXVector4( 0.0, 0.0, 0.0 ) );
			materialResource.m_diffuse.SetColor(  IFXVector4( 1.0, 1.0, 1.0 ) );
			materialResource.m_specular.SetColor( IFXVector4( 0.0, 0.0, 0.0 ) );
			materialResource.m_emissive.SetColor( IFXVector4( 0.0, 0.0, 0.0 ) );
			materialResource.m_reflectivity = 0.1f;
			materialResource.m_opacity = 1.0f;
			pMaterialResources->AddResource( materialResource );
		}

		{
			TextureResourceList* pTextureResources = static_cast< TextureResourceList* >( Resources.GetResourceList( IDTF_TEXTURE ) );
			pTextureResources->SetType( IDTF_TEXTURE );
			Texture textureResource;
			textureResource.SetName( L"lines" );
			ImageFormat imageFormat;
			imageFormat.m_compressionType = IDTF_IMAGE_COMPRESSION_TYPE_PNG;
			imageFormat.m_alpha = IDTF_TRUE;
			imageFormat.m_blue = IDTF_TRUE;
			imageFormat.m_green = IDTF_TRUE;
			imageFormat.m_red = IDTF_TRUE;

			textureResource.AddImageFormat( imageFormat );
			textureResource.SetExternal( FALSE );
			textureResource.SetPath( L"lines2.tga" );
			textureResource.SetImageType( IDTF_IMAGE_TYPE_RGBA );

			const U8 testTexture1D[ 4 * 1 * 4 ] =
			{
				255, 0, 0, 255,		255, 0, 0, 200,		0, 0, 255, 100,		0, 0, 255,   0,
			};
			textureResource.m_textureImage.Initialize( 2, 1, 4 );
			textureResource.m_textureImage.SetData( testTexture1D );

			pTextureResources->AddResource( textureResource );
		}

		ModifierList& Modifiers = converter.m_modifierList;

		{
			ShadingModifier shadingModifier;
			shadingModifier.SetName( L"LineSet" );
			shadingModifier.SetType( IDTF_SHADING_MODIFIER );
			shadingModifier.SetChainType( IDTF_NODE );
			shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
			ShaderList shaderList;
			shaderList.AddShaderName( L"ModelShader1" );
			shadingModifier.AddShaderList( shaderList );
			Modifiers.AddModifier( &shadingModifier );
		}

	}

		// LineSet - test
		if( IFXSUCCESS(result) )
		{
			
			NodeList& Nodes = converter.m_nodeList;
			
			{
				ModelNode Model;
				Model.SetType( IDTF_MODEL );
				Model.SetName( L"LineSet" );
				Model.SetResourceName( L"LineSetModel" );
				ParentList Parents;
				ParentData Parent;
				Parent.SetParentName( L"<NULL>" );
				IFXMatrix4x4 Matrix;
				Matrix.Reset();
				Parent.SetParentTM( Matrix );
				Parents.AddParentData( Parent );
				Model.SetParentList( Parents );
				Nodes.AddNode( &Model );
			}
			
			SceneResources& Resources = converter.m_sceneResources;
			
			{
				const I32 lineCount = 1;
				ModelResourceList* pModelResources = static_cast< ModelResourceList* >( Resources.GetResourceList( IDTF_MODEL ) );
				pModelResources->SetType( IDTF_MODEL );
				LineSetResource lineSetResource;
				lineSetResource.SetName( L"LineSetModel" );
				lineSetResource.m_type = IDTF_LINE_SET;
				lineSetResource.lineCount = lineCount;
				lineSetResource.m_modelDescription.positionCount = 2;
				lineSetResource.m_modelDescription.basePositionCount = 0;
				lineSetResource.m_modelDescription.normalCount = 0;
				lineSetResource.m_modelDescription.diffuseColorCount = 0;
				lineSetResource.m_modelDescription.specularColorCount = 0;
				lineSetResource.m_modelDescription.textureCoordCount = 0;
				lineSetResource.m_modelDescription.boneCount = 0;
				lineSetResource.m_modelDescription.shadingCount = 1;
				ShadingDescription shadingDescription;
				shadingDescription.m_shaderId = 0;
				shadingDescription.m_textureLayerCount = 0;
				lineSetResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );
				I32 linePositionList [lineCount][2] = 
				{
					{ 0, 1 } 
				};
				for( int lineIndex = 0; lineIndex < lineCount; lineIndex++ )
					lineSetResource.m_linePositions.CreateNewElement().SetData( linePositionList[lineIndex][0], linePositionList[lineIndex][1] );
				I32 lineShadingList[lineCount] =
				{
					0
				};
				for( int lineIndex = 0; lineIndex < lineCount; lineIndex++ )
					lineSetResource.m_lineShaders.CreateNewElement() = lineShadingList[lineIndex];
				float positionList[2][3] =
				{
					{ 0, 0, 0 },
					{ 1, 0, 0 }
				};
				for( int positionIndex = 0; positionIndex < lineSetResource.m_modelDescription.positionCount; positionIndex++ )
					lineSetResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( positionList[positionIndex][0], positionList[positionIndex][1], positionList[positionIndex][2] ) );
				pModelResources->AddResource( &lineSetResource );
			}
			
			{
				ShaderResourceList* pShaderResources = static_cast< ShaderResourceList* >( Resources.GetResourceList( IDTF_SHADER ) );
				pShaderResources->SetType( IDTF_SHADER );
				Shader shaderResource;
				shaderResource.SetName( L"ModelShader1" );
				shaderResource.m_materialName = L"ModelMaterial";
				pShaderResources->AddResource( shaderResource );
			}
			
			{
				MaterialResourceList* pMaterialResources = static_cast< MaterialResourceList* >( Resources.GetResourceList( IDTF_MATERIAL ) );
				pMaterialResources->SetType( IDTF_MATERIAL );
				Material materialResource;
				materialResource.SetName( L"ModelMaterial" );
				materialResource.m_ambient.SetColor(  IFXVector4( 0.0, 0.0, 0.0 ) );
				materialResource.m_diffuse.SetColor(  IFXVector4( 1.0, 1.0, 1.0 ) );
				materialResource.m_specular.SetColor( IFXVector4( 0.0, 0.0, 0.0 ) );
				materialResource.m_emissive.SetColor( IFXVector4( 0.0, 0.0, 0.0 ) );
				materialResource.m_reflectivity = 0.1f;
				materialResource.m_opacity = 1.0;
				pMaterialResources->AddResource( materialResource );
			}
			
			
			ModifierList& Modifiers = converter.m_modifierList;
			
			{
				ShadingModifier shadingModifier;
				shadingModifier.SetName( L"LineSet" );
				shadingModifier.SetType( IDTF_SHADING_MODIFIER );
				shadingModifier.SetChainType( IDTF_NODE );
				shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
				ShaderList shaderList;
				shaderList.AddShaderName( L"ModelShader1" );
				shadingModifier.AddShaderList( shaderList );
				Modifiers.AddModifier( &shadingModifier );
			}
			
		}
	converter.Export( "test.idtf" );
	converter.Convert();

	//----------------------------------------------
	// Scene now built and in the U3D engine.
	// It is now time to examine the scene and/or
	// dump it to a debug file or a U3D file.
	//----------------------------------------------
	// Write out the scene to a U3D file if this is enabled.
	if ( IFXSUCCESS( result ) && ( fileOptions.exportOptions > 0 ) )
	{
		result = sceneUtils.WriteSceneToFile( fileOptions.outFile, fileOptions.exportOptions );
	}
	// If enabled, dump the scene to the debug file.
	if ( 0 && IFXSUCCESS( result ) && ( fileOptions.debugLevel > 0 ) )
	{
		U8 file[MAXIMUM_FILENAME_LENGTH];
		result = fileOptions.outFile.ConvertToRawU8( file, MAXIMUM_FILENAME_LENGTH );

		if ( IFXSUCCESS( result ) )
			result = sceneUtils.WriteDebugInfo( (const char*)file );
	}

	}
	IFXTRACE_GENERIC( L"[Converter] Exit code = %x\n", result);

	IFXRESULT comResult = IFXCOMUninitialize();
	IFXTRACE_GENERIC( L"[Converter] IFXCOMUninitialize %i\n", comResult );

	IFXDEBUG_SHUTDOWN();

	return result;
}

//***************************************************************************
//  Local functions
//***************************************************************************

