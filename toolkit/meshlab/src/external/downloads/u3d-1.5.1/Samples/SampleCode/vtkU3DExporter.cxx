#include "vtkU3DExporter.h"

#include "vtkActor2DCollection.h"
#include "vtkActor2D.h"
#include "vtkAssemblyPath.h"
#include "vtkCamera.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkDataSetMapper.h"
#include "vtkDataSetSurfaceFilter.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkGeometryFilter.h"
#include "vtkImageData.h"
#include "vtkLightCollection.h"
#include "vtkLight.h"
#include "vtkMath.h"
#include "vtkMergePoints.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper2D.h"
#include "vtkPolyDataMapper.h"
#include "vtkPolyDataNormals.h"
#include "vtkProperty.h"
#include "vtkRendererCollection.h"
#include "vtkRenderWindow.h"
#include "vtkSmartPointer.h"
#include "vtkTextActor.h"
#include "vtkTextProperty.h"
#include "vtkTexture.h"
#include "vtkTransform.h"
#include "vtkTriangleFilter.h"

#include <sstream>
#include <cassert>
#include <list>

#include "vtkStdString.h"

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

#define VTK_CREATE(type, var) \
  vtkSmartPointer<type> var = vtkSmartPointer<type>::New()

#define VTK_DECLARE(type, var) \
  vtkSmartPointer<type> var = NULL

#define VTK_NEW(type, var) \
                        var = vtkSmartPointer<type>::New()

#define sign(x) ((x<0.0) ? (-1.0) : (1.0))

class vtkPolyData;
// A hack to get to internal vtkMapper data
class VTKU3DEXPORTER_EXPORT vtkMyPolyDataMapper : public vtkPolyDataMapper
{
public:
  vtkFloatArray *GetColorCoordinates();
  vtkImageData *GetColorTextureMap();
  vtkUnsignedCharArray  *GetColors();
};
vtkFloatArray *vtkMyPolyDataMapper::GetColorCoordinates()
{
  return this->ColorCoordinates;
}
vtkImageData *vtkMyPolyDataMapper::GetColorTextureMap()
{
  return this->ColorTextureMap;
}
vtkUnsignedCharArray  *vtkMyPolyDataMapper::GetColors()
{
  return this->Colors;
}

using namespace U3D_IDTF;

// forward declarations
static bool vtkU3DExporterWriterUsingCellColors(vtkActor* anActor);

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkU3DExporter);

//----------------------------------------------------------------------------
vtkU3DExporter::vtkU3DExporter()
{
  this->FileName = NULL;
  this->MeshCompression = 0;
}
//----------------------------------------------------------------------------
vtkU3DExporter::~vtkU3DExporter()
{
  this->SetFileName(0);
}

struct u3dLine
{
  vtkIdType point1;
  vtkIdType point2;
  double color[3];
};

typedef std::list<u3dLine>  u3dLineSet;

static void AddLine(u3dLineSet& LineSet,
                    vtkIdType point1, vtkIdType point2,
                    const unsigned char *color1, const unsigned char *color2)
{
  u3dLine Line;
  Line.point1 = (point1 <= point2) ? point1 : point2;
  Line.point2 = (point1 <= point2) ? point2 : point1;
  if (color1)
  {
    if (color2)
    {
      Line.color[0] = (color1[0] + color2[0])/510.0;
      Line.color[1] = (color1[1] + color2[1])/510.0;
      Line.color[2] = (color1[2] + color2[2])/510.0;
    }
    else
    {
      Line.color[0] = color1[0]/255.0;
      Line.color[1] = color1[1]/255.0;
      Line.color[2] = color1[2]/255.0;
    }
  }
  else
  {
    Line.color[0] = 0.0;
    Line.color[1] = 0.0;
    Line.color[2] = 0.0;
  }
  for (u3dLineSet::iterator it = LineSet.begin(); it != LineSet.end(); ++it)
  {
    if ((*it).point1 == Line.point1 && (*it).point2 == Line.point2)
    {
      (*it).color[0] = (Line.color[0] + (*it).color[0])/2.0;
      (*it).color[1] = (Line.color[1] + (*it).color[1])/2.0;
      (*it).color[2] = (Line.color[2] + (*it).color[2])/2.0;
      return;
    }
  }
  LineSet.push_back(Line);
}

void CreateMaterial( SceneConverter& converter, vtkActor* anActor, bool emissive, Material& materialResource )
{
  double tempd[3];
  double tempf2;

  SceneResources& Resources = converter.m_sceneResources;
  MaterialResourceList* pMaterialResources = static_cast< MaterialResourceList* >( Resources.GetResourceList( IDTF_MATERIAL ) );
  vtkProperty* prop = anActor->GetProperty();

  wchar_t materialName[256];
  swprintf(materialName, 255, L"Material%u", pMaterialResources->GetResourceCount());

  materialResource.SetName( materialName );

  tempf2 = prop->GetAmbient();
  prop->GetAmbientColor(tempd);
  tempd[0]*=tempf2;
  tempd[1]*=tempf2;
  tempd[2]*=tempf2;
  materialResource.m_ambient.SetColor(  IFXVector4( tempd[0], tempd[1], tempd[2] ) );
  // Set diffuse color
  tempf2 = prop->GetDiffuse();
  prop->GetDiffuseColor(tempd);
  tempd[0]*=tempf2;
  tempd[1]*=tempf2;
  tempd[2]*=tempf2;
  materialResource.m_diffuse.SetColor(  IFXVector4( tempd[0], tempd[1], tempd[2] ) );
  tempf2 = prop->GetSpecular();
  prop->GetSpecularColor(tempd);
  tempd[0]*=tempf2;
  tempd[1]*=tempf2;
  tempd[2]*=tempf2;
  // Set specular color
  materialResource.m_specular.SetColor( IFXVector4( tempd[0], tempd[1], tempd[2] ) );
  if (emissive)
  {
    tempf2 = prop->GetAmbient();
    prop->GetAmbientColor(tempd);
    tempd[0]*=tempf2;
    tempd[1]*=tempf2;
    tempd[2]*=tempf2;
  }
  else
  {
    tempd[0] = tempd[1] = tempd[2] = 0.0f;
  }
  materialResource.m_emissive.SetColor( IFXVector4( tempd[0], tempd[1], tempd[2] ) );
  // Material shininess
  materialResource.m_reflectivity = prop->GetSpecularPower()/128.0;
  // Material transparency
  materialResource.m_opacity = prop->GetOpacity();

}

void CreateModelNode( vtkActor* anActor, const wchar_t* name, ModelNode& Model )
{
       Model.SetType( IDTF_MODEL );
       Model.SetName( name );
       Model.SetResourceName( name );
       ParentList Parents;
       ParentData Parent;
       Parent.SetParentName( L"<NULL>" );
       IFXMatrix4x4 Matrix;
       VTK_CREATE(vtkMatrix4x4, tempm);
       anActor->vtkProp3D::GetMatrix(tempm);
       Matrix[ 0] = tempm->Element[0][0];
       Matrix[ 1] = tempm->Element[1][0];
       Matrix[ 2] = tempm->Element[2][0];
       Matrix[ 3] = tempm->Element[3][0];
       Matrix[ 4] = tempm->Element[0][1];
       Matrix[ 5] = tempm->Element[1][1];
       Matrix[ 6] = tempm->Element[2][1];
       Matrix[ 7] = tempm->Element[3][1];
       Matrix[ 8] = tempm->Element[0][2];
       Matrix[ 9] = tempm->Element[1][2];
       Matrix[10] = tempm->Element[2][2];
       Matrix[11] = tempm->Element[3][2];
       Matrix[12] = tempm->Element[0][3];
       Matrix[13] = tempm->Element[1][3];
       Matrix[14] = tempm->Element[2][3];
       Matrix[15] = tempm->Element[3][3];
       Parent.SetParentTM( Matrix );
       Parents.AddParentData( Parent );
       Model.SetParentList( Parents );
}
//----------------------------------------------------------------------------
void vtkU3DExporter::WriteData()
{
  vtkRenderer *ren;
  vtkActorCollection *ac;
//  vtkActor2DCollection *a2Dc;
  vtkActor *anActor, *aPart;
//  vtkActor2D *anTextActor2D, *aPart2D;
  vtkLightCollection *lc;
  vtkLight *aLight;
  vtkCamera *cam;

  // make sure the user specified a FileName or FilePointer
  if (this->FileName == NULL)
    {
    vtkErrorMacro(<< "Please specify FileName to use");
    return;
    }

  // Let's assume the first renderer is the right one
  // first make sure there is only one renderer in this rendering window
  //if (this->RenderWindow->GetRenderers()->GetNumberOfItems() > 1)
  //  {
  //  vtkErrorMacro(<< "U3D files only support one renderer per window.");
  //  return;
  //  }

  // get the renderer
  ren = this->RenderWindow->GetRenderers()->GetFirstRenderer();

  // make sure it has at least one actor
  if (ren->GetActors()->GetNumberOfItems() < 1)
    {
    vtkErrorMacro(<< "no actors found for writing U3D file.");
    return;
    }

  //
  //  Write header
  //
  vtkDebugMacro("Writing U3D file");

  IFXRESULT result = IFX_OK;

  result = IFXSetDefaultLocale();
  IFXTRACE_GENERIC(L"[Converter] IFXSetDefaultLocale %i\n", result);

  if( IFXSUCCESS(result) )
  {
    IFXDEBUG_STARTUP();
    result = IFXCOMInitialize();
  }

  if( !IFXSUCCESS(result) )
    return;

  {
  ConverterOptions converterOptions;
  FileOptions fileOptions;

  wchar_t *wU3DFileName = new wchar_t [mbstowcs(NULL, this->FileName, 32000)+1+4];
  mbstowcs(wU3DFileName, this->FileName, 32000);
  wcsncat(wU3DFileName,L".u3d",4);
  fileOptions.outFile    = wU3DFileName;
  delete [] wU3DFileName;

  fileOptions.exportOptions  = IFXExportOptions(65535);
  fileOptions.profile    = 0;
  fileOptions.scalingFactor  = 1.0f;
  fileOptions.debugLevel    = 1;

  converterOptions.positionQuality = 1000;
  converterOptions.texCoordQuality = 1000;
  converterOptions.normalQuality   = 1000;
  converterOptions.diffuseQuality  = 1000;
  converterOptions.specularQuality = 1000;
  converterOptions.geoQuality      = 1000;
  converterOptions.textureQuality  = 100;
  converterOptions.animQuality     = 1000;
  converterOptions.textureLimit    = 0;
  converterOptions.removeZeroAreaFaces  = TRUE;
  converterOptions.zeroAreaFaceTolerance  = 100.0f * FLT_EPSILON;
  converterOptions.excludeNormals  = FALSE;

  SceneUtilities sceneUtils;

  result = sceneUtils.InitializeScene( fileOptions.profile, fileOptions.scalingFactor );


  SceneConverter converter( &sceneUtils, &converterOptions );

  if( IFXSUCCESS(result) )
  {{
  NodeList&       Nodes     = converter.m_nodeList;
  SceneResources& Resources = converter.m_sceneResources;
  ModifierList&   Modifiers = converter.m_modifierList;

  Resources.GetResourceList( IDTF_LIGHT    )->SetType( IDTF_LIGHT    );
  Resources.GetResourceList( IDTF_MODEL    )->SetType( IDTF_MODEL    );
  Resources.GetResourceList( IDTF_VIEW     )->SetType( IDTF_VIEW     );
  Resources.GetResourceList( IDTF_SHADER   )->SetType( IDTF_SHADER   );
  Resources.GetResourceList( IDTF_MATERIAL )->SetType( IDTF_MATERIAL );
  Resources.GetResourceList( IDTF_TEXTURE  )->SetType( IDTF_TEXTURE  );

  // Start write the Camera
  {
    {
      ViewResourceList* pViewResources = static_cast< ViewResourceList* >( Resources.GetResourceList( IDTF_VIEW ) );
      ViewResource defaultViewResource;
      defaultViewResource.SetName( L"SceneViewResource" );
      defaultViewResource.AddRootNode( L"" );
      pViewResources->AddResource( defaultViewResource );
    }

  cam = ren->GetActiveCamera();
    {
      ViewNode View;
      View.SetType( IDTF_VIEW );
      View.SetName( L"DefaultView" );
      View.SetResourceName( L"SceneViewResource" );
      ParentList Parents;
      ParentData Parent;
      Parent.SetParentName( L"<NULL>" );
      IFXMatrix4x4 Matrix;
      VTK_CREATE(vtkMatrix4x4, tempm);
      tempm->DeepCopy(cam->GetViewTransformMatrix());
      tempm->Invert();
      Matrix[ 0] = tempm->Element[0][0];
      Matrix[ 1] = tempm->Element[1][0];
      Matrix[ 2] = tempm->Element[2][0];
      Matrix[ 3] = tempm->Element[3][0];
      Matrix[ 4] = tempm->Element[0][1];
      Matrix[ 5] = tempm->Element[1][1];
      Matrix[ 6] = tempm->Element[2][1];
      Matrix[ 7] = tempm->Element[3][1];
      Matrix[ 8] = tempm->Element[0][2];
      Matrix[ 9] = tempm->Element[1][2];
      Matrix[10] = tempm->Element[2][2];
      Matrix[11] = tempm->Element[3][2];
      Matrix[12] = tempm->Element[0][3];
      Matrix[13] = tempm->Element[1][3];
      Matrix[14] = tempm->Element[2][3];
      Matrix[15] = tempm->Element[3][3];
      Parent.SetParentTM( Matrix );
      Parents.AddParentData( Parent );
      View.SetParentList( Parents );
      ViewNodeData ViewData;
      ViewData.SetUnitType( IDTF_VIEW_UNIT_PIXEL );
      ViewData.SetClipping( cam->GetClippingRange()[0], cam->GetClippingRange()[1] );
      ViewData.SetViewPort( VIEW_PORT_WIDTH, VIEW_PORT_HEIGHT, VIEW_PORT_H_POSITION, VIEW_PORT_V_POSITION );
      if ( cam->GetParallelProjection () )
        {
        ViewData.SetType( IDTF_ORTHO_VIEW );
        ViewData.SetProjection( static_cast<float>( cam->GetParallelScale() ) );
        }
      else
        {
        ViewData.SetType( IDTF_PERSPECTIVE_VIEW );
        ViewData.SetProjection( static_cast<float>( cam->GetViewAngle() ) );
        }

      View.SetViewData( ViewData );
      Nodes.AddNode( &View );
    }
  }
  // End of Camera

  // do the lights first the ambient then the others
  {
    LightResourceList* pLightResources = static_cast< LightResourceList* >( Resources.GetResourceList( IDTF_LIGHT ) );

    // ambient light
    {
      LightNode ambientLightNode;
      ambientLightNode.SetType( IDTF_LIGHT );
      ambientLightNode.SetName( L"AmbientLight" );
      ambientLightNode.SetResourceName( L"DefaultAmbientLight" );
      ParentList Parents;
      ParentData Parent;
      Parent.SetParentName( L"<NULL>" );
      IFXMatrix4x4 Matrix;
      Matrix.Reset();
      Parent.SetParentTM( Matrix );
      Parents.AddParentData( Parent );
      ambientLightNode.SetParentList( Parents );
      Nodes.AddNode( &ambientLightNode );

      LightResource ambientLightResource;
      ambientLightResource.SetName( L"DefaultAmbientLight" );
      ambientLightResource.m_type = IDTF_AMBIENT_LIGHT;
      ambientLightResource.m_color.SetColor( IFXVector4( ren->GetAmbient()[0], ren->GetAmbient()[1], ren->GetAmbient()[2] ) );
      ambientLightResource.m_attenuation.SetPoint( IFXVector3( 1.0f, 0.0f, 0.0f ) );
      ambientLightResource.m_intensity = 1.0f;
      ambientLightResource.m_spotAngle = 0.0f;
      pLightResources->AddResource( ambientLightResource );
    }

  lc = ren->GetLights();
  vtkCollectionSimpleIterator lsit;
  for (lc->InitTraversal(lsit); (aLight = lc->GetNextLight(lsit)); )
    {
    if (!aLight->LightTypeIsHeadlight() )
      {
  double *pos, *focus;
  double dir[3];
  IFXMatrix4x4 Matrix;
  Matrix.Reset();
  double a, b, c;

  LightNode lightNode;
  lightNode.SetType( IDTF_LIGHT );
  LightResource lightResource;

  pos = aLight->GetPosition();
  focus = aLight->GetFocalPoint();

  dir[0] = focus[0] - pos[0];
  dir[1] = focus[1] - pos[1];
  dir[2] = focus[2] - pos[2];
  vtkMath::Normalize(dir);
  a = -dir[0]; b = -dir[1]; c = -dir[2];

  if (aLight->GetPositional())
    {
    if (aLight->GetConeAngle() >= 180.0)
      {
      lightResource.m_type = IDTF_POINT_LIGHT;
      lightResource.m_spotAngle = 0.0f;
      Matrix[3*4 + 0] = pos[0];
      Matrix[3*4 + 1] = pos[1];
      Matrix[3*4 + 2] = pos[2];
      }
    else
      {
      lightResource.m_type = IDTF_SPOT_LIGHT;
      lightResource.m_spotAngle = aLight->GetConeAngle();
      if (sqrt(a*a+b*b) != 0.0)
        {
//            -b/sqrt(a*a+b*b)   -a/sqrt(a*a+b*b)           0.0
//          -a*c/sqrt(a*a+b*b) -b*c/sqrt(a*a+b*b) sqrt(a*a+b*b)
//                    a                  b             c
        Matrix[0*4 + 0] = -b/sqrt(a*a+b*b);
        Matrix[0*4 + 1] = -a/sqrt(a*a+b*b);
        Matrix[1*4 + 0] = -a*c/sqrt(a*a+b*b);
        Matrix[1*4 + 1] = -b*c/sqrt(a*a+b*b);
        Matrix[1*4 + 2] = sqrt(a*a+b*b);
        Matrix[2*4 + 0] = a;
        Matrix[2*4 + 1] = b;
        Matrix[2*4 + 2] = c;
        }
      else
        {
//          1 0 0
//          0 1 0
//          0 0 sign(c)
        Matrix[2*4 + 2] = sign(c);
        }
      Matrix[3*4 + 0] = pos[0];
      Matrix[3*4 + 1] = pos[1];
      Matrix[3*4 + 2] = pos[2];
      }
    }
  else
    {
    lightResource.m_type = IDTF_DIRECTIONAL_LIGHT;
    lightResource.m_spotAngle = 0.0f;
    if (sqrt(a*a+b*b) != 0.0)
      {
//            -b/sqrt(a*a+b*b)   -a/sqrt(a*a+b*b)           0.0
//          -a*c/sqrt(a*a+b*b) -b*c/sqrt(a*a+b*b) sqrt(a*a+b*b)
//                    a                  b             c
        Matrix[0*4 + 0] = -b/sqrt(a*a+b*b);
        Matrix[0*4 + 1] = -a/sqrt(a*a+b*b);
        Matrix[1*4 + 0] = -a*c/sqrt(a*a+b*b);
        Matrix[1*4 + 1] = -b*c/sqrt(a*a+b*b);
        Matrix[1*4 + 2] = sqrt(a*a+b*b);
        Matrix[2*4 + 0] = a;
        Matrix[2*4 + 1] = b;
        Matrix[2*4 + 2] = c;
      }
    else
      {
//          1 0 0
//          0 1 0
//          0 0 sign(c)
        Matrix[2*4 + 2] = sign(c);
      }
    }

    {                                                                                                       
    wchar_t lightName[128];
    swprintf(lightName, 127, L"Light%u", pLightResources->GetResourceCount());
    lightNode.SetName( lightName );
    lightNode.SetResourceName( lightName );
    lightResource.SetName( lightName );
    }
    ParentList Parents;
    ParentData Parent;
    Parent.SetParentName( L"<NULL>" );
    Parent.SetParentTM( Matrix );
    Parents.AddParentData( Parent );
    lightNode.SetParentList( Parents );
    Nodes.AddNode( &lightNode );
    lightResource.m_color.SetColor( IFXVector4( aLight->GetDiffuseColor()[0], aLight->GetDiffuseColor()[1], aLight->GetDiffuseColor()[2] ) );
    lightResource.m_attenuation.SetPoint( IFXVector3( aLight->GetAttenuationValues()[0], aLight->GetAttenuationValues()[0], aLight->GetAttenuationValues()[0] ) );
    lightResource.m_intensity = aLight->GetIntensity()*aLight->GetSwitch();
    pLightResources->AddResource( lightResource );

      }
    }

  }
  // End lights

  // do the actors now
  ModelResourceList*    pModelResources    = static_cast< ModelResourceList* >   ( Resources.GetResourceList( IDTF_MODEL ) );
  MaterialResourceList* pMaterialResources = static_cast< MaterialResourceList* >( Resources.GetResourceList( IDTF_MATERIAL ) );
  ShaderResourceList*   pShaderResources   = static_cast< ShaderResourceList* >  ( Resources.GetResourceList( IDTF_SHADER ) );
  TextureResourceList*  pTextureResources  = static_cast< TextureResourceList* > ( Resources.GetResourceList( IDTF_TEXTURE ) );
//  pModelResources->SetType( IDTF_MODEL );
//  pMaterialResources->SetType( IDTF_MATERIAL );
//  pShaderResources->SetType( IDTF_SHADER );
//  pTextureResources->SetType( IDTF_TEXTURE );


  ac = ren->GetActors();
  vtkAssemblyPath *apath;
  vtkCollectionSimpleIterator ait;
  for (ac->InitTraversal(ait); (anActor = ac->GetNextActor(ait)); )
    {
    for (anActor->InitPathTraversal(); (apath=anActor->GetNextPath()); )
      {
      if(anActor->GetVisibility()==0)
        continue;
      aPart=static_cast<vtkActor *>(apath->GetLastNode()->GetViewProp());
      // see if the actor has a mapper. it could be an assembly
      if (aPart->GetMapper() == NULL)
        continue;

      int isbm = aPart->GetMapper()->GetInterpolateScalarsBeforeMapping();

      // get the mappers input
      vtkDataSet *ds;
      ds = aPart->GetMapper()->GetInput();
      vtkPolyData *pd;
      vtkSmartPointer<vtkDataSetSurfaceFilter> gf;

      // we really want polydata
      if ( ds->GetDataObjectType() != VTK_POLY_DATA )
      {
        gf = vtkSmartPointer<vtkDataSetSurfaceFilter>::New();
        gf->SetInputData(ds);
        gf->Update();
        pd = gf->GetOutput();
      }
      else
      {
        pd = static_cast<vtkPolyData *>(ds);
      }

     if (pd->GetNumberOfCells() == 0 || pd->GetPoints()->GetNumberOfPoints() == 0)
       continue;


     {
     vtkProperty *prop;
     vtkSmartPointer<vtkTransform> trans;


     prop = aPart->GetProperty();

     int representation = prop->GetRepresentation();

     if (representation == VTK_POINTS)
       {
       // If representation is points, then we don't have to render different cell
       // types in separate shapes, since the cells type no longer matter.
       vtkPoints *points = pd->GetPoints();

       VTK_DECLARE(vtkPoints, upoints);
       vtkIdType numPoints = 0;
       {
       double bounds[6];
       points->ComputeBounds();
       points->GetBounds(bounds);
       VTK_NEW(vtkPoints, upoints);
       numPoints = points->GetNumberOfPoints();

       VTK_CREATE(vtkMergePoints, locator);
       locator->InitPointInsertion(upoints, bounds, numPoints);

       for (vtkIdType pid=0; pid < numPoints; pid++)
         {
         vtkIdType CurPoint;
         double* point = points->GetPoint(pid);
         locator->InsertUniquePoint(point, CurPoint);
         }
       numPoints = upoints->GetNumberOfPoints();
       }

       wchar_t name[256];
       swprintf(name, 255, L"Points%u", Nodes.GetNodeCount());

       // Create Node
       ModelNode Model;
       CreateModelNode(aPart, name, Model);
       Nodes.AddNode( &Model );

       PointSetResource pointSetResource;
       pointSetResource.SetName( name );
       pointSetResource.m_type = IDTF_POINT_SET;
       pointSetResource.pointCount = numPoints;
       pointSetResource.m_modelDescription.positionCount = numPoints;
       pointSetResource.m_modelDescription.basePositionCount = 0;
       pointSetResource.m_modelDescription.normalCount = 0;
       pointSetResource.m_modelDescription.diffuseColorCount = 0;
       pointSetResource.m_modelDescription.specularColorCount = 0;
       pointSetResource.m_modelDescription.textureCoordCount = 0;
       pointSetResource.m_modelDescription.boneCount = 0;
       pointSetResource.m_modelDescription.shadingCount = 1;
       ShadingDescription shadingDescription;
       shadingDescription.m_shaderId = 0;
       shadingDescription.m_textureLayerCount = 0;
       pointSetResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );

       for (vtkIdType pid=0; pid < numPoints; pid++)
       {
         pointSetResource.m_pointPositions.CreateNewElement() = pid;
       }

       for (vtkIdType pid=0; pid < numPoints; pid++)
       {
         pointSetResource.m_pointShaders.CreateNewElement() = 0;
       }

       for (vtkIdType pid=0; pid < numPoints; pid++)
       {
         const double* point = upoints->GetPoint(pid);
         pointSetResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( point[0], point[1], point[2] ) );
       }

       pModelResources->AddResource( &pointSetResource );

       ShadingModifier shadingModifier;
       shadingModifier.SetName( name );
       shadingModifier.SetType( IDTF_SHADING_MODIFIER );
       shadingModifier.SetChainType( IDTF_NODE );
       shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
       ShaderList shaderList;

       // Create a material, although it does not affect the displayed result
       // use emissive to color points
       Material materialResource;
       CreateMaterial(converter, aPart, true, materialResource);
       pMaterialResources->AddResource( materialResource );

       wchar_t shaderName[256];
       swprintf(shaderName, 255, L"Shader%u", pShaderResources->GetResourceCount());
       Shader shaderResource;
       shaderResource.SetName( shaderName );
       shaderResource.m_materialName = materialResource.GetName();
       shaderResource.m_useVertexColor = IDTF_FALSE;
       pShaderResources->AddResource( shaderResource );

       shaderList.AddShaderName( shaderName );
       shadingModifier.AddShaderList( shaderList );
       Modifiers.AddModifier( &shadingModifier );
       }
     else
       {
       // When rendering as lines or surface, we need to respect the cell
       // structure. This requires rendering polys/tstrips, lines, verts in
       // separate shapes.
       vtkIdType numVerts  = pd->GetNumberOfVerts();
       vtkIdType numLines  = pd->GetNumberOfLines();
       vtkIdType numPolys  = pd->GetNumberOfPolys();
       vtkIdType numStrips = pd->GetNumberOfStrips();

       if ((numPolys > 0 || numStrips > 0) && representation == VTK_SURFACE)
         {
         vtkPolyData *ppd;
         vtkPointData *pntData;
         vtkCellData *cellData;
         vtkPoints *points;
         vtkDataArray *normals;
         vtkUnsignedCharArray *colors = NULL;

         // we really want triangle polydata
         VTK_CREATE(vtkTriangleFilter, tg);
         tg->SetInputData(pd);
         tg->PassVertsOff();
         tg->PassLinesOff();
         tg->Update();
         ppd = tg->GetOutput();

         points = ppd->GetPoints();
         pntData = ppd->GetPointData();
         cellData = ppd->GetCellData();

         vtkCellArray* polys = ppd->GetPolys();
         vtkIdType numPolys = polys->GetNumberOfCells();

         if (numPolys == 0)
           continue;

         VTK_CREATE(vtkActor, myActor);
         VTK_CREATE(vtkPolyDataMapper, myPolyDataMapper);
         myActor->ShallowCopy(aPart);
         if (myActor->GetMapper()->IsA("vtkPolyDataMapper"))
           myPolyDataMapper->ShallowCopy(myActor->GetMapper());
         else if (myActor->GetMapper()->IsA("vtkDataSetMapper"))
           myPolyDataMapper->ShallowCopy(static_cast<vtkDataSetMapper*>(myActor->GetMapper())->GetPolyDataMapper());
         else
           continue;
         myPolyDataMapper->SetInputData(ppd);
         myPolyDataMapper->SetInterpolateScalarsBeforeMapping(isbm);
         myActor->SetMapper(myPolyDataMapper);

         colors  = myActor->GetMapper()->MapScalars(255.0);

         vtkMyPolyDataMapper *myMyPolyDataMapper = static_cast<vtkMyPolyDataMapper*>(myPolyDataMapper.GetPointer());
         vtkFloatArray *colorCoordinates = myMyPolyDataMapper->GetColorCoordinates();
         vtkImageData *colorTextureMap = myMyPolyDataMapper->GetColorTextureMap();

         // Are we using cell colors.
         bool cell_colors = vtkU3DExporterWriterUsingCellColors(myActor);

         normals = pntData->GetNormals();

         // Are we using cell normals.
         bool cell_normals = false;
         if (prop->GetInterpolation() == VTK_FLAT || !normals)
           {
           // use cell normals, if any.
           normals = cellData->GetNormals();
           cell_normals = true;
           }

         VTK_CREATE(vtkPoints, upoints); // unique points
         VTK_CREATE(vtkIdList, upointIds); //  ids of original points in upoints
         vtkIdType numPoints = points->GetNumberOfPoints();

         {
         double pointbounds[6];
         points->ComputeBounds();
         points->GetBounds(pointbounds);
         VTK_CREATE(vtkMergePoints, pointlocator);
         pointlocator->InitPointInsertion(upoints, pointbounds, numPoints);

         for (vtkIdType pid=0; pid < numPoints; pid++)
           {
           vtkIdType CurPoint;
           double* point = points->GetPoint(pid);
           pointlocator->InsertUniquePoint(point, CurPoint);
           upointIds->InsertNextId(CurPoint);
           }
         numPoints = upoints->GetNumberOfPoints();
         }

         VTK_DECLARE(vtkPoints, unormals);   // same thing with normals
         VTK_DECLARE(vtkIdList, unormalIds);
         vtkIdType numNormals = 0;

         if (normals)
           {
           double normalbounds[6] = { -1, 1, -1, 1, -1, 1};
           VTK_NEW(vtkPoints, unormals);
           VTK_NEW(vtkIdList, unormalIds);
           numNormals = normals->GetNumberOfTuples();

           VTK_CREATE(vtkMergePoints, normallocator);
           normallocator->InitPointInsertion(unormals, normalbounds, numNormals);

           for (vtkIdType pid=0; pid < numNormals; pid++)
             {
             vtkIdType CurNormal;
             double* normal = normals->GetTuple(pid);
             double length = vtkMath::Norm(normal);
             if (length != 0.0)
               {
               for (int j=0; j < 3; j++)
                 {
                 normal[j] = normal[j] / length;
                 }
               }
             normallocator->InsertUniquePoint(normal, CurNormal);
             unormalIds->InsertNextId(CurNormal);
             }
           numNormals = unormals->GetNumberOfPoints();
           }

           wchar_t name[256];
           swprintf(name, 255, L"Mesh%u", Nodes.GetNodeCount());

           // Create Node
           ModelNode Model;
           CreateModelNode(aPart, name, Model);
           Model.SetVisibility( L"BOTH" );
           Nodes.AddNode( &Model );

           // Create Resource, Materials, Shaders and Modifier
           if ( colorCoordinates )
             {
             vtkIdType npts = 0;
             vtkIdType *indx = 0;

             VTK_DECLARE(vtkPoints, ucolorCoordinates);   // unique colors
             VTK_DECLARE(vtkIdList, ucolorCoordinatesIds); // ids of original colors in ucolors
             vtkIdType numColorCoordinates = 0;

             {
             double colorCoordinatesBounds[6] = { 0, 0, 0, 0, 0, 0 };
             colorCoordinates->GetRange(colorCoordinatesBounds, 0);
             VTK_NEW(vtkPoints, ucolorCoordinates);
             VTK_NEW(vtkIdList, ucolorCoordinatesIds);
             numColorCoordinates = colorCoordinates->GetNumberOfTuples();

             VTK_CREATE(vtkMergePoints, colorCoordinatesLocator);
             colorCoordinatesLocator->InitPointInsertion(ucolorCoordinates, colorCoordinatesBounds, numColorCoordinates);

             for (vtkIdType id=0; id < numColorCoordinates; id++)
               {
               vtkIdType colorCoordinateId;
               double colorPoint[3] = { 0, 0, 0 };
               colorPoint[0] = colorCoordinates->GetValue(id);
               colorCoordinatesLocator->InsertUniquePoint(colorPoint, colorCoordinateId);
               ucolorCoordinatesIds->InsertNextId(colorCoordinateId);
               }
             numColorCoordinates = ucolorCoordinates->GetNumberOfPoints();
             }

             MeshResource meshResource;
             meshResource.SetName( name );
             meshResource.m_type = IDTF_MESH;
             meshResource.faceCount = numPolys;
             meshResource.m_modelDescription.positionCount = numPoints;
             meshResource.m_modelDescription.basePositionCount = (MeshCompression ? 0 : numPoints);
             meshResource.m_modelDescription.normalCount = numNormals;
             meshResource.m_modelDescription.diffuseColorCount = 0;
             meshResource.m_modelDescription.specularColorCount = 0;
             meshResource.m_modelDescription.textureCoordCount = numColorCoordinates;
             meshResource.m_modelDescription.boneCount = 0;
             meshResource.m_modelDescription.shadingCount = 1;
             ShadingDescription shadingDescription;
             shadingDescription.m_shaderId = 0;
             shadingDescription.m_textureLayerCount = 1;
             shadingDescription.AddTextureCoordDimension( 1 );
             meshResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );

             for (polys->InitTraversal(); polys->GetNextCell(npts,indx); )
               {
               meshResource.m_facePositions.CreateNewElement().SetData( upointIds->GetId(indx[0]), upointIds->GetId(indx[1]), upointIds->GetId(indx[2]) );
               }

             if (normals)
               {
               if (cell_normals)
                 {
                 for (vtkIdType pid=0; pid < numPolys; pid++)
                   {
                   meshResource.m_faceNormals.CreateNewElement().SetData( unormalIds->GetId(pid), unormalIds->GetId(pid), unormalIds->GetId(pid) );
                   }
                 }
               else
                 {
                 for (polys->InitTraversal(); polys->GetNextCell(npts,indx); )
                   {
                   meshResource.m_faceNormals.CreateNewElement().SetData( unormalIds->GetId(indx[0]), unormalIds->GetId(indx[1]), unormalIds->GetId(indx[2]) );
                   }
                 }
               }

             for (vtkIdType pid=0; pid < numPolys; pid++)
               {
               meshResource.m_faceShaders.CreateNewElement() = 0;
               }

             for (polys->InitTraversal(); polys->GetNextCell(npts,indx); )
               {
               meshResource.m_faceTextureCoords.CreateNewElement().m_texCoords.CreateNewElement().SetData( ucolorCoordinatesIds->GetId(indx[0]), ucolorCoordinatesIds->GetId(indx[1]), ucolorCoordinatesIds->GetId(indx[2]) );
               }

             for (vtkIdType pid=0; pid < numPoints; pid++)
               {
               const double* point = upoints->GetPoint(pid);
               meshResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( point[0], point[1], point[2] ) );
               }

             for (vtkIdType pid=0; pid < numNormals; pid++)
               {
               const double* normal = unormals->GetPoint(pid);
               meshResource.m_normals.CreateNewElement().SetPoint( IFXVector3 ( normal[0], normal[1], normal[2] ) );
               }

             for (vtkIdType pid=0; pid < numColorCoordinates; pid++)
               {
               meshResource.m_textureCoords.CreateNewElement().Set( (ucolorCoordinates->GetPoint(pid))[0], 0, 0, 0 );
               }

             if (!MeshCompression)
               {
               for (vtkIdType pid=0; pid < numPoints; pid++)
                 {
                 meshResource.m_basePositions.CreateNewElement() = pid;
                 }
               }

             pModelResources->AddResource( &meshResource );

             ShadingModifier shadingModifier;
             shadingModifier.SetName( name );
             shadingModifier.SetType( IDTF_SHADING_MODIFIER );
             shadingModifier.SetChainType( IDTF_NODE );
             shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
             ShaderList shaderList;

             Material materialResource;
             CreateMaterial(converter, myActor, false, materialResource);
             pMaterialResources->AddResource( materialResource );

             wchar_t textureName[256];
             swprintf(textureName, 255, L"Texture%u", pTextureResources->GetResourceCount());
             {
             Texture textureResource;
             textureResource.SetName( textureName );
             ImageFormat imageFormat;
             imageFormat.m_compressionType = IDTF_IMAGE_COMPRESSION_TYPE_PNG;
             imageFormat.m_alpha = IDTF_TRUE;
             imageFormat.m_blue = IDTF_TRUE;
             imageFormat.m_green = IDTF_TRUE;
             imageFormat.m_red = IDTF_TRUE;

             wchar_t texturePath[512];
             swprintf(texturePath, 511, L"%s_%ls.tga", this->FileName, textureName);
             textureResource.AddImageFormat( imageFormat );
             textureResource.SetExternal( FALSE );
             textureResource.SetPath( texturePath );
             textureResource.SetImageType( IDTF_IMAGE_TYPE_RGBA );

             int extent[6];
             colorTextureMap->GetExtent(extent);
             const int colorTextureMapSize = extent[1]+1;
             U8* internalColorTexture = new U8[ colorTextureMapSize * 1 * 4 ];
             for (int i=0; i < colorTextureMapSize; i++)
               {
               const unsigned char *color = static_cast<unsigned char *>(colorTextureMap->GetScalarPointer(i, 0, 0));
               internalColorTexture[i*4+0] = color[0];
               internalColorTexture[i*4+1] = color[1];
               internalColorTexture[i*4+2] = color[2];
               internalColorTexture[i*4+3] = color[3];
               }
             textureResource.m_textureImage.Initialize( colorTextureMapSize, 1, 4 );
             textureResource.m_textureImage.SetData( internalColorTexture );
             pTextureResources->AddResource( textureResource );
             delete[] internalColorTexture;
             }

             wchar_t shaderName[256];
             swprintf(shaderName, 255, L"Shader%u", pShaderResources->GetResourceCount());
             {
             Shader shaderResource;
             shaderResource.SetName( shaderName );
             shaderResource.m_materialName = materialResource.GetName();
             shaderResource.m_useVertexColor = IDTF_FALSE;
             TextureLayer textureLayer;
             textureLayer.m_channel = 0;
             textureLayer.m_intensity = 1.0f;
             textureLayer.m_blendFunction = L"REPLACE";
             textureLayer.m_blendSource = L"ALPHA";
             textureLayer.m_blendConstant = 1.0;
             textureLayer.m_alphaEnabled = IDTF_TRUE;
             textureLayer.m_repeat = L"NONE";
             textureLayer.m_textureName = textureName;
             shaderResource.AddTextureLayer( textureLayer );
             pShaderResources->AddResource( shaderResource );
             }

             shaderList.AddShaderName( shaderName );
             shadingModifier.AddShaderList( shaderList );
             Modifiers.AddModifier( &shadingModifier );
             }
           else if ( colors && cell_colors)
             {
             vtkIdType npts = 0;
             vtkIdType *indx = 0;

             VTK_DECLARE(vtkDoubleArray, ucolors);   // unique colors
             VTK_DECLARE(vtkIdList, ucolorIds); // ids of original colors in ucolors
             vtkIdType numColors = 0;
             // create unique colors list, should really use bsearch or something less naive.
             if (colors)
               {
               VTK_CREATE(vtkUnsignedCharArray, uucolors);   // unique colors
               uucolors->SetNumberOfComponents(4);
               VTK_NEW(vtkIdList, ucolorIds);
               numColors = colors->GetNumberOfTuples();

               for (vtkIdType pid=0; pid < numColors; pid++)
                 {
                 vtkIdType curColor = uucolors->GetNumberOfTuples();
                 unsigned char color[4];
                 colors->GetTupleValue(pid, color);
                 for (vtkIdType cid=0; cid < uucolors->GetNumberOfTuples(); cid++)
                   {
                   unsigned char uucolor[4];
                   uucolors->GetTupleValue(cid, uucolor);
                   if (color[0] == uucolor[0] && color[1] == uucolor[1] && color[2] == uucolor[2] && color[3] == uucolor[3])
                     {
                     curColor = cid;
                     break;
                     }
                   }
                 if (curColor == uucolors->GetNumberOfTuples())
                   curColor = uucolors->InsertNextTupleValue(color);
                 ucolorIds->InsertNextId(curColor);
                 }
               numColors = uucolors->GetNumberOfTuples();
               VTK_NEW(vtkDoubleArray, ucolors);
               ucolors->SetNumberOfComponents(4);
               for (vtkIdType cid=0; cid < uucolors->GetNumberOfTuples(); cid++)
                 {
                 unsigned char uucolor[4];
                 uucolors->GetTupleValue(cid, uucolor);
                 const double ucolor[4] = { uucolor[0]/255.0, uucolor[1]/255.0, uucolor[2]/255.0, uucolor[3]/255.0 };
                 ucolors->InsertNextTupleValue(ucolor);
                 }
               }

             MeshResource meshResource;
             meshResource.SetName( name );
             meshResource.m_type = IDTF_MESH;
             meshResource.faceCount = numPolys;
             meshResource.m_modelDescription.positionCount = numPoints;
             meshResource.m_modelDescription.basePositionCount = (MeshCompression ? 0 : numPoints);
             meshResource.m_modelDescription.normalCount = numNormals;
             meshResource.m_modelDescription.diffuseColorCount = 0;
             meshResource.m_modelDescription.specularColorCount = 0;
             meshResource.m_modelDescription.textureCoordCount = 0;
             meshResource.m_modelDescription.boneCount = 0;
             if (colors)
               {
               meshResource.m_modelDescription.shadingCount = numColors;
               for (vtkIdType pid=0; pid < numColors; pid++)
                 {
                     ShadingDescription shadingDescription;
                     shadingDescription.m_shaderId = pid;
                     shadingDescription.m_textureLayerCount = 0;
                     meshResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );
                 }
               }
             else
               {
               meshResource.m_modelDescription.shadingCount = 1;
               ShadingDescription shadingDescription;
               shadingDescription.m_shaderId = 0;
               shadingDescription.m_textureLayerCount = 0;
               meshResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );
               }

             for (polys->InitTraversal(); polys->GetNextCell(npts,indx); )
               {
               meshResource.m_facePositions.CreateNewElement().SetData( upointIds->GetId(indx[0]), upointIds->GetId(indx[1]), upointIds->GetId(indx[2]) );
               }

             if (normals)
               {
               if (cell_normals)
                 {
                 for (vtkIdType pid=0; pid < numPolys; pid++)
                   {
                   meshResource.m_faceNormals.CreateNewElement().SetData( unormalIds->GetId(pid), unormalIds->GetId(pid), unormalIds->GetId(pid) );
                   }
                 }
               else
                 {
                 for (polys->InitTraversal(); polys->GetNextCell(npts,indx); )
                   {
                   meshResource.m_faceNormals.CreateNewElement().SetData( unormalIds->GetId(indx[0]), unormalIds->GetId(indx[1]), unormalIds->GetId(indx[2]) );
                   }
                 }
               }

             for (vtkIdType pid=0; pid < numPolys; pid++)
               {
               meshResource.m_faceShaders.CreateNewElement() = (colors ? ucolorIds->GetId(pid) : 0);
               }

             for (vtkIdType pid=0; pid < numPoints; pid++)
               {
               const double* point = upoints->GetPoint(pid);
               meshResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( point[0], point[1], point[2] ) );
               }

             for (vtkIdType pid=0; pid < numNormals; pid++)
               {
               const double* normal = unormals->GetPoint(pid);
               meshResource.m_normals.CreateNewElement().SetPoint( IFXVector3 ( normal[0], normal[1], normal[2] ) );
               }

             if (!MeshCompression)
               {
               for (vtkIdType pid=0; pid < numPoints; pid++)
                 {
                 meshResource.m_basePositions.CreateNewElement() = pid;
                 }
               }

             pModelResources->AddResource( &meshResource );

             ShadingModifier shadingModifier;
             shadingModifier.SetName( name );
             shadingModifier.SetType( IDTF_SHADING_MODIFIER );
             shadingModifier.SetChainType( IDTF_NODE );
             shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );

             if (colors)
               {
               for (vtkIdType pid=0; pid < numColors; pid++)
                 {
                 double color[4];
                 ucolors->GetTupleValue(pid, color);
                 double intens;
                 wchar_t materialName[256];
                 swprintf(materialName, 255, L"Material%u", pMaterialResources->GetResourceCount());

                 Material materialResource;
                 materialResource.SetName( materialName );
                 intens = prop->GetAmbient();
                 materialResource.m_ambient.SetColor(  IFXVector4( intens*color[0], intens*color[1], intens*color[2] ) );
                 intens = prop->GetDiffuse();
                 materialResource.m_diffuse.SetColor(  IFXVector4( intens*color[0], intens*color[1], intens*color[2] ) );
                 intens = prop->GetSpecular();
                 materialResource.m_specular.SetColor( IFXVector4( intens*color[0], intens*color[1], intens*color[2] ) );
                 intens = prop->GetAmbient();
                 materialResource.m_emissive.SetColor( IFXVector4( intens*color[0], intens*color[1], intens*color[2] ) );
                 materialResource.m_reflectivity = prop->GetSpecularPower()/128.0;
                 // Material transparency
                 // materialResource.m_opacity = prop->GetOpacity();
                 materialResource.m_opacity = color[3];
                 pMaterialResources->AddResource( materialResource );

                 wchar_t shaderName[256];
                 swprintf(shaderName, 255, L"Shader%u", pShaderResources->GetResourceCount());
                 Shader shaderResource;
                 shaderResource.SetName( shaderName );
                 shaderResource.m_materialName = materialName;
                 shaderResource.m_useVertexColor = IDTF_FALSE;
                 pShaderResources->AddResource( shaderResource );

                 ShaderList shaderList;
                 shaderList.AddShaderName( shaderName );
                 shadingModifier.AddShaderList( shaderList );
                 }
               }
             else
               {
               Material materialResource;
               CreateMaterial(converter, myActor, false, materialResource);
               pMaterialResources->AddResource( materialResource );

               wchar_t shaderName[256];
               swprintf(shaderName, 255, L"Shader%u", pShaderResources->GetResourceCount());
               Shader shaderResource;
               shaderResource.SetName( shaderName );
               shaderResource.m_materialName = materialResource.GetName();
               shaderResource.m_useVertexColor = IDTF_FALSE;
               pShaderResources->AddResource( shaderResource );

               ShaderList shaderList;
               shaderList.AddShaderName( shaderName );
               shadingModifier.AddShaderList( shaderList );
               }
             Modifiers.AddModifier( &shadingModifier );
             }
           else
           // Use vertex colors, lighting and transparency are not supported by Acrobat <= 9.1
             {
             vtkIdType npts = 0;
             vtkIdType *indx = 0;

             VTK_DECLARE(vtkPoints, ucolors);   // unique colors
             VTK_DECLARE(vtkIdList, ucolorIds); // ids of original colors in ucolors
             vtkIdType numColors = 0;

             if (colors)
               {
               double colorbounds[6] = { 0, 1, 0, 1, 0, 1};
               VTK_NEW(vtkPoints, ucolors);
               VTK_NEW(vtkIdList, ucolorIds);
               numColors = colors->GetNumberOfTuples();

               VTK_CREATE(vtkMergePoints, colorlocator);
               colorlocator->InitPointInsertion(ucolors, colorbounds, numColors);

               for (vtkIdType pid=0; pid < numColors; pid++)
                 {
                 vtkIdType CurColor;
                 unsigned char color[4];
                 colors->GetTupleValue(pid, color);
                 double dcolor[3];
                 dcolor[0] = color[0]/255.0;
                 dcolor[1] = color[1]/255.0;
                 dcolor[2] = color[2]/255.0;
                 colorlocator->InsertUniquePoint(dcolor, CurColor);
                 ucolorIds->InsertNextId(CurColor);
                 }
               numColors = ucolors->GetNumberOfPoints();
               }

               MeshResource meshResource;
               meshResource.SetName( name );
               meshResource.m_type = IDTF_MESH;
               meshResource.faceCount = numPolys;
               meshResource.m_modelDescription.positionCount = numPoints;
               meshResource.m_modelDescription.basePositionCount = (MeshCompression ? 0 : numPoints);
               meshResource.m_modelDescription.normalCount = numNormals;
               meshResource.m_modelDescription.diffuseColorCount = numColors;
               meshResource.m_modelDescription.specularColorCount = 0;
               meshResource.m_modelDescription.textureCoordCount = 0;
               meshResource.m_modelDescription.boneCount = 0;
               meshResource.m_modelDescription.shadingCount = 1;
               ShadingDescription shadingDescription;
               shadingDescription.m_shaderId = 0;
               shadingDescription.m_textureLayerCount = 0;
               meshResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );

               for (polys->InitTraversal(); polys->GetNextCell(npts,indx); )
                 {
                 meshResource.m_facePositions.CreateNewElement().SetData( upointIds->GetId(indx[0]), upointIds->GetId(indx[1]), upointIds->GetId(indx[2]) );
                 }

               if (normals)
                 {
                 if (cell_normals)
                   {
                   for (vtkIdType pid=0; pid < numPolys; pid++)
                     {
                     meshResource.m_faceNormals.CreateNewElement().SetData( unormalIds->GetId(pid), unormalIds->GetId(pid), unormalIds->GetId(pid) );
                     }
                   }
                 else
                   {
                   for (polys->InitTraversal(); polys->GetNextCell(npts,indx); )
                     {
                     meshResource.m_faceNormals.CreateNewElement().SetData( unormalIds->GetId(indx[0]), unormalIds->GetId(indx[1]), unormalIds->GetId(indx[2]) );
                     }
                   }
                 }

               for (vtkIdType pid=0; pid < numPolys; pid++)
                 {
                 meshResource.m_faceShaders.CreateNewElement() = 0;
                 }

               if (colors)
                 {
                 if (cell_colors) // Redundant now, but if Adobe fixes things in v10 it may be used
                   {
                     for (vtkIdType pid=0; pid < numPolys; pid++)
                       {
                       meshResource.m_faceDiffuseColors.CreateNewElement().SetData( ucolorIds->GetId(pid), ucolorIds->GetId(pid), ucolorIds->GetId(pid));
                       }
                   }
                 else
                   {
                   for (polys->InitTraversal(); polys->GetNextCell(npts,indx); )
                     {
                     meshResource.m_faceDiffuseColors.CreateNewElement().SetData( ucolorIds->GetId(indx[0]), ucolorIds->GetId(indx[1]), ucolorIds->GetId(indx[2]));
                     }
                   }
                 }

               for (vtkIdType pid=0; pid < numPoints; pid++)
                 {
                 const double* point = upoints->GetPoint(pid);
                 meshResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( point[0], point[1], point[2] ) );
                 }

               for (vtkIdType pid=0; pid < numNormals; pid++)
                 {
                 const double* normal = unormals->GetPoint(pid);
                 meshResource.m_normals.CreateNewElement().SetPoint( IFXVector3 ( normal[0], normal[1], normal[2] ) );
                 }

               for (vtkIdType pid=0; pid < numColors; pid++)
                 {
                 double* color = ucolors->GetPoint(pid);
                 meshResource.m_diffuseColors.CreateNewElement().SetColor( IFXVector4 ( color[0], color[1], color[2] ) );
                 }

               if (!MeshCompression)
                 {
                 for (vtkIdType pid=0; pid < numPoints; pid++)
                   {
                   meshResource.m_basePositions.CreateNewElement() = pid;
                   }
                 }

               pModelResources->AddResource( &meshResource );

               ShadingModifier shadingModifier;
               shadingModifier.SetName( name );
               shadingModifier.SetType( IDTF_SHADING_MODIFIER );
               shadingModifier.SetChainType( IDTF_NODE );
               shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
               ShaderList shaderList;

               if (colors)
                 {
                 Material materialResource;
                 CreateMaterial(converter, myActor, false, materialResource);
                 materialResource.m_diffuse.SetColor(  IFXVector4( 1, 1, 1 ) );
                 pMaterialResources->AddResource( materialResource );

                 wchar_t shaderName[256];
                 swprintf(shaderName, 255, L"Shader%u", pShaderResources->GetResourceCount());
                 Shader shaderResource;
                 shaderResource.SetName( shaderName );
                 shaderResource.m_materialName = materialResource.GetName();
                 shaderResource.m_useVertexColor = IDTF_TRUE;
                 pShaderResources->AddResource( shaderResource );

                 shaderList.AddShaderName( shaderName );
                 }
                 else
                 {
                 Material materialResource;
                 CreateMaterial(converter, myActor, false, materialResource);
                 pMaterialResources->AddResource( materialResource );

                 wchar_t shaderName[256];
                 swprintf(shaderName, 255, L"Shader%u", pShaderResources->GetResourceCount());
                 Shader shaderResource;
                 shaderResource.SetName( shaderName );
                 shaderResource.m_materialName = materialResource.GetName();
                 shaderResource.m_useVertexColor = IDTF_FALSE;
                 pShaderResources->AddResource( shaderResource );

                 shaderList.AddShaderName( shaderName );
                 }
               shadingModifier.AddShaderList( shaderList );
               Modifiers.AddModifier( &shadingModifier );
               }

         }

        // Lines rendering reflects limitation of Acrobat <= 9.1
        // The only way to color lines isto create a material for every line color used
        // Textures and vertex colors are not supported
        if ((numLines > 0) || ((numPolys > 0 || numStrips > 0) && representation == VTK_WIREFRAME))
          {
          vtkPointData *pntData;
          vtkCellData *cellData;
          vtkPoints *points;
          vtkProperty *prop;
          vtkUnsignedCharArray *colors = NULL;

          points = pd->GetPoints();
          pntData = pd->GetPointData();
          cellData = pd->GetCellData();

          VTK_CREATE(vtkActor, myActor);
          VTK_DECLARE(vtkPolyDataMapper, myPolyDataMapper);
          VTK_DECLARE(vtkDataSetMapper, myDataSetMapper);
          myActor->ShallowCopy(aPart);
          if (myActor->GetMapper()->IsA("vtkPolyDataMapper"))
            {
            VTK_NEW(vtkPolyDataMapper, myPolyDataMapper);
            myPolyDataMapper->ShallowCopy(myActor->GetMapper());
            myPolyDataMapper->SetInputData(pd);
            myActor->SetMapper(myPolyDataMapper);
            }
          else if (myActor->GetMapper()->IsA("vtkDataSetMapper"))
            {
            VTK_NEW(vtkDataSetMapper, myDataSetMapper);
            myDataSetMapper->ShallowCopy(myActor->GetMapper());
            myDataSetMapper->SetInputData(pd);
            myActor->SetMapper(myDataSetMapper);
            }
          else
            continue;
          // Essential to turn of interpolate scalars otherwise GetScalars() may return NULL.
          myActor->GetMapper()->SetInterpolateScalarsBeforeMapping(0);

          colors  = myActor->GetMapper()->MapScalars(255.0);

          // Are we using cell colors.
          bool cell_colors = vtkU3DExporterWriterUsingCellColors(myActor);

          prop = myActor->GetProperty();
          int representation = prop->GetRepresentation();

          vtkCellArray* lines = pd->GetLines();
          vtkCellArray* polys = pd->GetPolys();
          vtkCellArray* tstrips = pd->GetStrips();

          VTK_CREATE(vtkPoints, upoints);
          VTK_CREATE(vtkIdList, upointIds);
          vtkIdType numPoints = points->GetNumberOfPoints();

          {
            double pointbounds[6];
            points->ComputeBounds();
            points->GetBounds(pointbounds);
            VTK_CREATE(vtkMergePoints, pointlocator);
            pointlocator->InitPointInsertion(upoints, pointbounds, numPoints);

            for (vtkIdType pid=0; pid < numPoints; pid++)
            {
              vtkIdType CurPoint;
              double* point = points->GetPoint(pid);
              pointlocator->InsertUniquePoint(point, CurPoint);
              upointIds->InsertNextId(CurPoint);
            }
            numPoints = upoints->GetNumberOfPoints();
          }

          vtkIdType  lnumLines = 0;                   // number of lines
          u3dLineSet LineSet;

#define mAddLine(id1, id2)                           \
{                                                    \
vtkIdType point1 = upointIds->GetId(indx[id1]);      \
vtkIdType point2 = upointIds->GetId(indx[id2]);      \
if (colors)                                          \
{                                                    \
if (cell_colors)                                     \
{                                                    \
unsigned char color[4];                              \
colors->GetTupleValue(cellOffset, color);            \
AddLine(LineSet, point1, point2, color, NULL);       \
}                                                    \
else                                                 \
{                                                    \
unsigned char color1[4];                             \
colors->GetTupleValue(indx[id1], color1);            \
unsigned char color2[4];                             \
colors->GetTupleValue(indx[id2], color2);            \
AddLine(LineSet, point1, point2, color1, color2);    \
}                                                    \
}                                                    \
else                                                 \
{                                                    \
AddLine(LineSet, point1, point2, NULL, NULL);        \
}                                                    \
}

          if (numPolys > 0 && representation == VTK_WIREFRAME)
          {
            vtkIdType cellOffset = numVerts+numLines;
            vtkIdType npts = 0;
            vtkIdType *indx = 0;
            for (polys->InitTraversal(); polys->GetNextCell(npts,indx); cellOffset++)
            {
              for (vtkIdType cc=0; cc < npts; cc++)
              {
                mAddLine(cc, (cc+1)%npts);
              }
            }
          }

          if (numLines > 0)
          {
            vtkIdType cellOffset = numVerts;
            vtkIdType npts = 0;
            vtkIdType *indx = 0;
            for (lines->InitTraversal(); lines->GetNextCell(npts,indx); cellOffset++)
            {
              if (npts == 1)
              {
                mAddLine(0, 0);
              }
              for (vtkIdType cc=0; cc < npts-1; cc++)
              {
                mAddLine(cc, cc+1);
              }
            }
          }

          if (numStrips > 0 && representation == VTK_WIREFRAME)
          {
            vtkIdType cellOffset = numVerts+numLines+numPolys;
            vtkIdType npts = 0;
            vtkIdType *indx = 0;
            for (tstrips->InitTraversal(); tstrips->GetNextCell(npts,indx); cellOffset++)
            {
              if (npts == 1)
              {
                mAddLine(0, 0);
              }
              if (npts  > 1)
              {
                mAddLine(0, 1);
              }
              for (vtkIdType cc=2; cc < npts; cc++)
              {
                mAddLine(cc-1, cc);
                mAddLine(cc-2, cc);
              }
            }
          }
#undef mAddLine

          lnumLines = LineSet.size();

          VTK_CREATE(vtkPoints, lucolors);       // unique colors used in lines
          VTK_CREATE(vtkIdList, lucolorIds);     // indexes in the above list of line colors - 1 per line
          vtkIdType  lnumColors = 0;             // number of unique colors used in lines

          if (colors)
          {
            double colorbounds[6] = { 0, 1, 0, 1, 0, 1};

            VTK_CREATE(vtkMergePoints, lcolorlocator);
            lcolorlocator->InitPointInsertion(lucolors, colorbounds, lnumLines);

            for (u3dLineSet::iterator it = LineSet.begin(); it != LineSet.end(); ++it)
            {
              vtkIdType CurColor;
              lcolorlocator->InsertUniquePoint((*it).color, CurColor);
              lucolorIds->InsertNextId(CurColor);
            }
            lnumColors = lucolors->GetNumberOfPoints();
          }

          VTK_CREATE(vtkIdList, lupointIds);   // indexes of unique points used in lines
          VTK_CREATE(vtkIdList, lupointIdIds); // indexes in the above list of line points - 2 per line
          vtkIdType  lnumPoints = 0;           // number of unique points used in lines
          for (u3dLineSet::iterator it = LineSet.begin(); it != LineSet.end(); ++it)
          {
            lupointIdIds->InsertNextId(lupointIds->InsertUniqueId((*it).point1));
            lupointIdIds->InsertNextId(lupointIds->InsertUniqueId((*it).point2));
          }
          lnumPoints = lupointIds->GetNumberOfIds();

          wchar_t name[256];
          swprintf(name, 255, L"Lines%u", Nodes.GetNodeCount());

          // Create Node
          ModelNode Model;
          CreateModelNode(aPart, name, Model);
          Nodes.AddNode( &Model );

          LineSetResource lineSetResource;
          lineSetResource.SetName( name );
          lineSetResource.m_type = IDTF_LINE_SET;
          lineSetResource.lineCount = lnumLines;
          lineSetResource.m_modelDescription.positionCount = lnumPoints;
          lineSetResource.m_modelDescription.basePositionCount = 0;
          lineSetResource.m_modelDescription.normalCount = 0;
          lineSetResource.m_modelDescription.diffuseColorCount = 0;
          lineSetResource.m_modelDescription.specularColorCount = 0;
          lineSetResource.m_modelDescription.textureCoordCount = 0;
          lineSetResource.m_modelDescription.boneCount = 0;
          lineSetResource.m_modelDescription.shadingCount = (colors ? lnumColors : 1);
          for (vtkIdType pid=0; pid < (colors ? lnumColors : 1); pid++)
          {
            ShadingDescription shadingDescription;
            shadingDescription.m_shaderId = pid;
            shadingDescription.m_textureLayerCount = 0;
            lineSetResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );
          }

          for (vtkIdType pid=0; pid < lnumLines; pid++)
          {
            lineSetResource.m_linePositions.CreateNewElement().SetData( lupointIdIds->GetId(pid*2+0), lupointIdIds->GetId(pid*2+1) );
          }

          for (vtkIdType pid=0; pid < lnumLines; pid++)
          {
            lineSetResource.m_lineShaders.CreateNewElement() = (colors ? lucolorIds->GetId(pid) : 0);
          }

          for (vtkIdType pid=0; pid < lnumPoints; pid++)
          {
            const double* point = upoints->GetPoint(lupointIds->GetId(pid));
            lineSetResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( point[0], point[1], point[2] ) );
          }

          pModelResources->AddResource( &lineSetResource );

          ShadingModifier shadingModifier;
          shadingModifier.SetName( name );
          shadingModifier.SetType( IDTF_SHADING_MODIFIER );
          shadingModifier.SetChainType( IDTF_NODE );
          shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
          ShaderList shaderList;

          if (colors)
          {
            for (vtkIdType pid=0; pid < lnumColors; pid++)
            {
              const double* color = lucolors->GetPoint(pid);
              wchar_t materialName[256];
              swprintf(materialName, 255, L"Material%u", pMaterialResources->GetResourceCount());

              Material materialResource;
              materialResource.SetName( materialName );
              materialResource.m_ambient.SetColor(  IFXVector4( color[0], color[1], color[2] ) );
              materialResource.m_diffuse.SetColor(  IFXVector4( color[0], color[1], color[2] ) );
              materialResource.m_specular.SetColor( IFXVector4( color[0], color[1], color[2] ) );
              materialResource.m_emissive.SetColor( IFXVector4( color[0], color[1], color[2] ) );
              materialResource.m_reflectivity = 0.1;
              materialResource.m_opacity = 1.0;
              pMaterialResources->AddResource( materialResource );

              wchar_t shaderName[256];
              swprintf(shaderName, 255, L"Shader%u", pShaderResources->GetResourceCount());
              Shader shaderResource;
              shaderResource.SetName( shaderName );
              shaderResource.m_materialName = materialName;
              shaderResource.m_useVertexColor = IDTF_FALSE;
              pShaderResources->AddResource( shaderResource );

              shaderList.AddShaderName( shaderName );
            }
          }
          else
          {
            Material materialResource;
            CreateMaterial(converter, myActor, true, materialResource);
            pMaterialResources->AddResource( materialResource );

            wchar_t shaderName[256];
            swprintf(shaderName, 255, L"Shader%u", pShaderResources->GetResourceCount());
            Shader shaderResource;
            shaderResource.SetName( shaderName );
            shaderResource.m_materialName = materialResource.GetName();
            shaderResource.m_useVertexColor = IDTF_FALSE;
            pShaderResources->AddResource( shaderResource );

            shaderList.AddShaderName( shaderName );
          }
          shadingModifier.AddShaderList( shaderList );
          Modifiers.AddModifier( &shadingModifier );

          }

        // Reflects a limitation of Acrobat <= 9.1 that points have no color assigned
        // May change in the future if they fix Acrobat or show me the way colors can be applied
        if (numVerts > 0)
          {
          vtkPoints *points  = pd->GetPoints();
          vtkCellArray* verts = pd->GetVerts();

          VTK_CREATE(vtkPoints, upoints);
          vtkIdType numPoints = points->GetNumberOfPoints();

          {
          double pointbounds[6];
          points->ComputeBounds();
          points->GetBounds(pointbounds);
          VTK_CREATE(vtkMergePoints, pointlocator);
          pointlocator->InitPointInsertion(upoints, pointbounds, numPoints);

          vtkIdType npts = 0;
          vtkIdType *indx = 0;
          for (verts->InitTraversal(); verts->GetNextCell(npts,indx); )
            {
            for (vtkIdType cc=0; cc < npts; cc++)
              {
              vtkIdType CurPoint;
              double* point = points->GetPoint(indx[cc]);
              pointlocator->InsertUniquePoint(point, CurPoint);
              }
            }
          numPoints = upoints->GetNumberOfPoints();
          }

          wchar_t name[256];
          swprintf(name, 255, L"Points%u", Nodes.GetNodeCount());

          // Create Node
          ModelNode Model;
          CreateModelNode(aPart, name, Model);
          Nodes.AddNode( &Model );

          PointSetResource pointSetResource;
          pointSetResource.SetName( name );
          pointSetResource.m_type = IDTF_POINT_SET;
          pointSetResource.pointCount = numPoints;
          pointSetResource.m_modelDescription.positionCount = numPoints;
          pointSetResource.m_modelDescription.basePositionCount = 0;
          pointSetResource.m_modelDescription.normalCount = 0;
          pointSetResource.m_modelDescription.diffuseColorCount = 0;
          pointSetResource.m_modelDescription.specularColorCount = 0;
          pointSetResource.m_modelDescription.textureCoordCount = 0;
          pointSetResource.m_modelDescription.boneCount = 0;
          pointSetResource.m_modelDescription.shadingCount = 1;
          ShadingDescription shadingDescription;
          shadingDescription.m_shaderId = 0;
          shadingDescription.m_textureLayerCount = 0;
          pointSetResource.m_shadingDescriptions.AddShadingDescription( shadingDescription );

          for (vtkIdType pid=0; pid < numPoints; pid++)
          {
            pointSetResource.m_pointPositions.CreateNewElement() = pid;
          }

          for (vtkIdType pid=0; pid < numPoints; pid++)
          {
            pointSetResource.m_pointShaders.CreateNewElement() = 0;
          }

          for (vtkIdType pid=0; pid < numPoints; pid++)
          {
            const double* point = upoints->GetPoint(pid);
            pointSetResource.m_positions.CreateNewElement().SetPoint( IFXVector3 ( point[0], point[1], point[2] ) );
          }

          pModelResources->AddResource( &pointSetResource );

          ShadingModifier shadingModifier;
          shadingModifier.SetName( name );
          shadingModifier.SetType( IDTF_SHADING_MODIFIER );
          shadingModifier.SetChainType( IDTF_NODE );
          shadingModifier.SetAttributes( ATTRMESH | ATTRLINE | ATTRPOINT | ATTRGLYPH );
          ShaderList shaderList;

          // Create a material, although it does not affect the displayed result
          // use emissive to color points
          Material materialResource;
          CreateMaterial(converter, aPart, true, materialResource);
          pMaterialResources->AddResource( materialResource );

          wchar_t shaderName[256];
          swprintf(shaderName, 255, L"Shader%u", pShaderResources->GetResourceCount());
          Shader shaderResource;
          shaderResource.SetName( shaderName );
          shaderResource.m_materialName = materialResource.GetName();
          shaderResource.m_useVertexColor = IDTF_FALSE;
          pShaderResources->AddResource( shaderResource );

          shaderList.AddShaderName( shaderName );
          shadingModifier.AddShaderList( shaderList );
          Modifiers.AddModifier( &shadingModifier );

        }
      }
    }
      }
    }
  }
  char *idtfFileName = new char [strlen(this->FileName)+1+5];
  strcpy(idtfFileName, this->FileName);
  strcat(idtfFileName,".idtf");
  converter.Export( idtfFileName );
  delete [] idtfFileName;
  converter.Convert();
  }

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
  if ( IFXSUCCESS( result ) && ( fileOptions.debugLevel > 0 ) )
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

}

//----------------------------------------------------------------------------
static bool vtkU3DExporterWriterUsingCellColors(vtkActor* anActor)
{
  int cellFlag = 0;
  vtkMapper* mapper = anActor->GetMapper();
  vtkAbstractMapper::GetScalars(
                                mapper->GetInput(),
                                mapper->GetScalarMode(),
                                mapper->GetArrayAccessMode(),
                                mapper->GetArrayId(),
                                mapper->GetArrayName(), cellFlag);
  return (cellFlag == 1);
}
