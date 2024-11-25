#if defined(_MSC_VER) 
// disable warning "'QtConcurrent::BlockSizeManager' : assignment operator could not be generated"
#pragma warning( disable : 4512 )
#endif

#include "JavaScriptBuilder.h"

#include <QScriptEngine>
#include <QMessageBox>
#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <QThread>
#include "SyntopiaCore/Logging/Logging.h"
#include "../../SyntopiaCore/GLEngine/Sphere.h"
#include "../../SyntopiaCore/GLEngine/Raytracer/RayTracer.h"
#include "../../StructureSynth/Model/Rendering/OpenGLRenderer.h"
#include "../../StructureSynth/Parser/Tokenizer.h"
#include "../../StructureSynth/Parser/Preprocessor.h"
#include "../../StructureSynth/Model/RuleSet.h"
#include "../../StructureSynth/Model/Builder.h"
#include "../../StructureSynth/Parser/EisenParser.h"
#include "../../SyntopiaCore/Exceptions/Exception.h"

using namespace SyntopiaCore::Logging;
using namespace SyntopiaCore::GLEngine;
using namespace StructureSynth::Model;
using namespace StructureSynth::Parser;
using namespace SyntopiaCore::Exceptions;
using namespace StructureSynth::Model::Rendering;


namespace StructureSynth {
	namespace JavaScriptSupport {	

		namespace {
			// Dont know why Qt has chosen to make 'sleep' protected.
			class MyThread : public QThread {
			public:
				static void sleep(unsigned long msecs) { msleep(msecs); }
			};
		};



		void World::setColor2(Vector3 rgb, float alpha) {
			this->rgb = rgb.getObj();
			this->alpha = alpha;
		}

		void World::addSphere2(Vector3 center, float radius) {
			SyntopiaCore::GLEngine::Object3D* o = new SyntopiaCore::GLEngine::Sphere( center.getObj(), radius);
			o->setColor(this->rgb, this->alpha);
			engine->addObject(o);
		}



		Vector3::Vector3(){ };

		Vector3::Vector3(float x, float y, float z){ 
			//INFO(QString("Vector3(%1,%2,%3)").arg(x).arg(y).arg(z));
			v = SyntopiaCore::Math::Vector3f(x,y,z); 
		};

		Vector3::Vector3(const StructureSynth::JavaScriptSupport::Vector3 & vx) : QObject(), QScriptable() {
			v = vx.v;
			//INFO(QString("Vector3 CopyConstructor(%1,%2,%3)").arg(v.x()).arg(v.y()).arg(v.z()));

		}

		Builder::Builder(StructureSynth::GUI::MainWindow* mainWindow, SyntopiaCore::GLEngine::EngineWidget* engine3D, QString dir) : mainWindow(mainWindow), engine3D(engine3D) {
				width = engine3D->width();
				height = engine3D->height();
				workingDir = dir;
			};

		void Builder::build() {

			engine3D->setDisabled(true);


			Rendering::OpenGLRenderer renderTarget(engine3D);
			renderTarget.begin(); // we clear before parsing...

			Preprocessor pp;
			QString out = pp.Process(loadedSystem);
			Tokenizer tokenizer(out);
			EisenParser e(&tokenizer);
			RuleSet* rs = e.parseRuleset();
			rs->resolveNames();
			Model::Builder b(&renderTarget, rs, false);
			b.build();
			if (b.wasCancelled()) throw Exception("User cancelled");

			renderTarget.end();
			engine3D->setRaytracerCommands(b.getRaytracerCommands());
			//INFO(QString("Setting %1 raytracer commands.").arg(raytracerCommands.count()));
			delete(rs);
			rs = 0;

			engine3D->setDisabled(false);
			engine3D->requireRedraw();

		}

		void Builder::load(QString fileName) {	
			fileName = QDir(workingDir).absoluteFilePath(fileName);
			QFile file(fileName);
			if (!file.open(QFile::ReadOnly | QFile::Text)) {
				throw Exception(QString("Cannot read file %1: %2.").arg(fileName).arg(file.errorString()));
			} else {
				QTextStream in(&file);
				loadedSystem = in.readAll();
				originalSystem = loadedSystem;
			}
		};

		void Builder::prepend(QString prescript) {
			loadedSystem = prescript + "\n" + loadedSystem;
		}

		void Builder::append(QString postscript) {
			loadedSystem = loadedSystem + "\n" + postscript;
		}

		void Builder::define(QString input, QString value) {
			QStringList s = loadedSystem.split("\n");
			for (int i = 0; i < s.count(); i++) {

				if (!s[i].contains("#define", Qt::CaseInsensitive)) {
					//INFO("N: " + s[i]);
					s[i] = s[i].replace(input, value);
				} else {
					//INFO("X: " + s[i]);
				}
			}
			loadedSystem = s.join("\n");

		};

		void Builder::renderToFile(QString fileName, bool overwrite) {
			fileName = QDir(workingDir).absoluteFilePath(fileName);
			
			build();
			engine3D->requireRedraw();
			engine3D->update();
			qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
			engine3D->update();
			qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

			QImage image = engine3D->grabFrameBuffer();

			QFileInfo fi(fileName);
			if (fi.exists()) {
				if (!overwrite) {
					if (QMessageBox::question(0, "File Error", "Overwrite file: " + fi.absoluteFilePath() + "?", 
						QMessageBox::Ok | QMessageBox::Cancel, QMessageBox::Cancel) == QMessageBox::Cancel) {
							WARNING("Cancelled save.");
							return;

					}
				}
				INFO("Overwriting: " + fi.absoluteFilePath());		
			}
			bool succes = image.save(fileName);
			if (succes) {
				INFO("Saved: " + fi.absoluteFilePath());		
			} else {
				WARNING("Failed to save: " + fi.absoluteFilePath());
			}
		};

		void Builder::reset() { loadedSystem = originalSystem; };

		void Builder::templateRenderToFile(QString templateName, QString fileName, bool overwrite) {
			fileName = QDir(workingDir).absoluteFilePath(fileName);
			if (!overwrite && QFileInfo(fileName).exists()) {
				throw Exception("File already exists: " + fileName + ". Set 'overwrite' argument to true or use another filename.");
			}
			QDir d(mainWindow->getTemplateDir());
			QString templateFileName = d.absoluteFilePath(templateName);
			INFO("Starting Template Renderer: " + fileName);
			try {
				QFile file(templateFileName);
				Template myTemplate(file);
				mainWindow->templateRender(fileName, &myTemplate, loadedSystem, width, height); 
			} catch (Exception& er) {
				WARNING(er.getMessage());
			}
		}

		/// Execute a process.
		void Builder::execute(QString fileName, QString args, bool waitForFinish) {
			QProcess p;

			// Replace enviroment settings
			QStringList env = QProcess::systemEnvironment();
			foreach (QString es, env) {
				QStringList l = es.split("=");
				if (l.count() == 2) {
					fileName = fileName.replace("%"+l[0]+"%", l[1]);
					args = args.replace("%"+l[0]+"%", l[1]);
				}
			}
			fileName.replace("\"", "");

			QString dir = QFileInfo(fileName).absolutePath();
			INFO("Working Directory: " + dir);
			INFO("Command: " + fileName);
   			INFO("Args: " + args);
   
			// Split arguments. Be sure to respect quotes.
			bool inQuote = false;
			QStringList out;
			QString buffer;
			for (int i = 0; i < args.size(); ++i) {
				if (args.at(i) == QLatin1Char('"')) {
					inQuote = !inQuote;
					continue;
				}

				if (args.at(i) == ' ' && !inQuote) {
					if (!buffer.isEmpty()) out.append(buffer);
					buffer = "";
					continue;
				}
				buffer += args.at(i);	
			}
			if (!buffer.isEmpty())  out.append(buffer);
			//for (int i = 0; i < out.count(); i++) INFO("args:" + out[i]);
				
			// Finally start the process...
			p.setWorkingDirectory(dir);
			p.start(fileName, out);
			if (!p.waitForStarted()) {
				throw Exception("Could not start process: " + QFileInfo(fileName).absoluteFilePath());
				return;
			}

			if (waitForFinish) {
				QProgressDialog progress("Executing "  + QFileInfo(fileName).absoluteFilePath(), "Abort", 0, 0, mainWindow);
				progress.setWindowModality(Qt::WindowModal);
				progress.setMinimumDuration(0);
				progress.show();
				QTime t = QTime::currentTime();
				while (p.state() == QProcess::Running) {
					qApp->processEvents();
					p.waitForFinished(100);
					progress.setLabelText("Executing "  + QFileInfo(fileName).absoluteFilePath() + QString("\n\nRunning for %1 seconds...").arg(
						t.secsTo(QTime::currentTime())));
					if (progress.wasCanceled()) {
						p.kill();
						throw Exception("User cancelled: " + QFileInfo(fileName).absoluteFilePath());		
						break;
					}
				}
				int secs = t.secsTo(QTime::currentTime());
				INFO("Executed "  + QFileInfo(fileName).absoluteFilePath() + QString("in %1 seconds...").arg(
						secs));
				//QString s = p.readAllStandardError();
				//QString s2 = p.readAllStandardOutput();
				
			}
		}


		void Builder::setSize(int w, int h) {
			if (w == 0 && h == 0) {
				width = engine3D->width();
				height = engine3D->height();
			} else if (w == 0) {
				height = h;
				width= (h*engine3D->width())/engine3D->height();
			} else if (h == 0) {
				width = w;
				height = (w*engine3D->height())/engine3D->width();
			} else {
				width = w;
				height = h;
			}	
		}

		void Builder::raytraceToFile(QString fileName, bool overwrite) {

			mainWindow->disableAllExcept(mainWindow->getProgressBox());
			fileName = QDir(workingDir).absoluteFilePath(fileName);
			
			RayTracer rt(engine3D,mainWindow->getProgressBox(),false);
			INFO(QString("Raytracing %1x%2 image...").arg(width).arg(height));
			QImage im = rt.calculateImage(width,height);
			mainWindow->enableAll();
			if (rt.wasCancelled()) throw Exception("User cancelled");
			QFileInfo fi(fileName);
			if (fi.exists()) {
				if (!overwrite) {
					if (QMessageBox::question(0, "File Error", "Overwrite file: " + fi.absoluteFilePath() + "?", 
						QMessageBox::Ok | QMessageBox::Cancel, QMessageBox::Cancel) == QMessageBox::Cancel) {

							WARNING("Cancelled save.");
							return;
					}
				}
				INFO("Overwriting: " + fi.absoluteFilePath());		
			}
			bool succes = im.save(fileName);
			if (succes) {
				INFO("Saved: " + fi.absoluteFilePath());		
			} else {
				WARNING("Failed to save: " + fi.absoluteFilePath());
			}

		}

	}
}
