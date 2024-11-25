#if defined(_MSC_VER) 
// disable warning "'QtConcurrent::BlockSizeManager' : assignment operator could not be generated"
#pragma warning( disable : 4512 )
#endif

#include "Debug.h"

#include <QScriptEngine>
#include <QMessageBox>
#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <QThread>
#include "SyntopiaCore/Logging/Logging.h"
#include "../../SyntopiaCore/Exceptions/Exception.h"
#include "../../SyntopiaCore/GLEngine/Sphere.h"

using namespace SyntopiaCore::Logging;
using namespace SyntopiaCore::Exceptions;


namespace StructureSynth {
	namespace JavaScriptSupport {	

		namespace {
			// Dont know why Qt has chosen to make 'sleep' protected.
			class MyThread : public QThread {
			public:
				static void sleep(unsigned long msecs) { msleep(msecs); }
			};


		};

		Debug::Debug(QStatusBar* statusBar) : statusBar(statusBar) {
			progress = 0;
		}

		Debug::~Debug() {
		}

		void Debug::Info(QString input) {
			INFO(input);
		}

		void Debug::Message(QString input) {
			QMessageBox::information(0, "JavaScript Message", input);
		}

		void Debug::ShowProgress(QString caption) {
			delete(progress); 
			progress = new QProgressDialog(caption, "Cancel", 0, 1000, 0);
			progress->setWindowModality(Qt::WindowModal);
			progress->show();

		}

		void Debug::SetProgress(double percentage) {
			if (progress) progress->setValue(percentage*1000.0);
			qApp->processEvents();
		}

		void Debug::HideProgress() {
			delete(progress);
			progress = 0;
		}

		void Debug::Sleep(int ms) {
			MyThread::sleep(ms);
		}

			
		void Debug::waitForMouseButton() {
			while (true) {

				statusBar->showMessage("Left Mousebutton to continue, right to quit.", 4000);
				if (QApplication::mouseButtons() == Qt::LeftButton) {
					//statusBar->showMessage("");
				
					break;
				} else if (QApplication::mouseButtons() == Qt::RightButton) {
					//statusBar->showMessage("");
					throw Exception("");
					break;
				}
				qApp->processEvents();
				//QThread::msleep(100);

			}
		}
	}
}
