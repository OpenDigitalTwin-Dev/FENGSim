#pragma once

#include <QString>
#include <QObject>
#include <QProgressDialog>
#include <QScriptable>
#include "../../SyntopiaCore/Math/Vector3.h"
#include "../../SyntopiaCore/GLEngine/EngineWidget.h"
#include "../GUI/MainWindow.h"

namespace StructureSynth {
	namespace JavaScriptSupport {	

		class Builder : public QObject {
			Q_OBJECT

		public:
			Builder(StructureSynth::GUI::MainWindow* mainWindow, SyntopiaCore::GLEngine::EngineWidget* engine3D, QString dir);
			~Builder() {};
			
		public slots:
			/// ----- These can be called from JavaScript -------

			/// Load an EisenScript system.
			void load(QString fileName);

			/// Simply does text substitutions (but ignores preprocessor lines!)
			void define(QString input, QString value);

			/// prepends the script with 'prescript'
			void prepend(QString prescript);

			/// prepends the script with 'prescript'
			void append(QString postscript);

			/// Build (OpenGL).
			void build();
			
			/// Render (OpenGL to file).
			void renderToFile(QString fileName, bool overwrite);
		
			/// Raytrace image with same dimensions as viewport to file.
			void raytraceToFile(QString fileName, bool overwrite);

			/// Raytrace image with same dimensions as viewport to file.
			void templateRenderToFile(QString templateName, QString fileName, bool overwrite);

			/// Execute a process.
			void execute(QString fileName, QString args, bool waitForFinish);

			/// Restores the original content (useful if substitutions have been made.)
			void reset();

			/// Sets the dimensions for the output (internal raytracer and templates).
			/// If both width and height is zero, the screen viewport size is used (default).
			/// If either width or height is zero, the other dimension is calculated 
			/// using the aspect ratio of the viewport.
			void setSize(int width, int height);
						
		private:
			StructureSynth::GUI::MainWindow* mainWindow;
			SyntopiaCore::GLEngine::EngineWidget* engine3D;
			QString loadedSystem;
			QString originalSystem;
			int width;
			int height;
			QString workingDir;
		};



		// Wrapper for the 3D vector object.
		class Vector3 : public QObject, protected QScriptable {
			Q_OBJECT

			Q_PROPERTY(float x READ readX WRITE writeX)
			Q_PROPERTY(float y READ readY WRITE writeY)
			Q_PROPERTY(float z READ readZ WRITE writeZ)
     
		public:
			Vector3();
			void operator=(const Vector3& rhs) { 
				writeX(rhs.readX()); 
				writeY(rhs.readY()); 
				writeZ(rhs.readZ());
			}

			Vector3(float x, float y, float z);
			Vector3(const StructureSynth::JavaScriptSupport::Vector3 & vx);
			
			float readX() const { return v.x(); }
			float readY() const { return v.y(); }
			float readZ() const { return v.z(); }
			void writeX(float v) { this->v.x() = v; }
			void writeY(float v) { this->v.y() = v; }
			void writeZ(float v) { this->v.z() = v; }
			
			SyntopiaCore::Math::Vector3f getObj() { return v; }
			
		public slots:
			QString toString() const { return QString("(%1,%2,%3)").arg(v.x()).arg(v.y()).arg(v.z()); };
			float length() const { return v.length(); };
			void add(const StructureSynth::JavaScriptSupport::Vector3& rhs) { 
				v.x() = v.x() + rhs.v.x();
				v.y() = v.y() + rhs.v.y();
				v.z() = v.z() + rhs.v.z();
			}

		private:
			SyntopiaCore::Math::Vector3f v;
		};

		
		class World : public QObject {
			Q_OBJECT

		public:
			
			World(SyntopiaCore::GLEngine::EngineWidget* engine) : 
				engine(engine), 
				rgb(SyntopiaCore::Math::Vector3f(1,0,0)),
				alpha(1.0f) {};

			SyntopiaCore::GLEngine::EngineWidget* getEngine() { return engine; }
			SyntopiaCore::Math::Vector3f getRgb() { return rgb; };
			float getAlpha() { return alpha; };
			
		public slots:
			void addSphere2(StructureSynth::JavaScriptSupport::Vector3 center, float radius);
			void setColor2(StructureSynth::JavaScriptSupport::Vector3 center, float alpha);
			void clear() { engine->clearWorld(); };
			
		private:
			QProgressDialog* progress;
			SyntopiaCore::GLEngine::EngineWidget* engine;
			SyntopiaCore::Math::Vector3f rgb;
			float alpha;
		};
		
	}
}

Q_DECLARE_METATYPE(StructureSynth::JavaScriptSupport::Vector3)






