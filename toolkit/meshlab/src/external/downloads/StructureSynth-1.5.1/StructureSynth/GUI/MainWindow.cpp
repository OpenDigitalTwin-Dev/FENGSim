
#include <QtGui>
#include <QDir>
#include <QClipboard>
#include <QDesktopServices>
#include <QImageWriter>
#include <QTextBlockUserData>
#include <QStack>
#include <QImage>
#include <QPixmap>
#include <QDialogButtonBox>

#include "MainWindow.h"
#include "VariableEditor.h"
#include "../Model/RandomStreams.h"
#include "../../SyntopiaCore/Logging/ListWidgetLogger.h"
#include "../../SyntopiaCore/Exceptions/Exception.h"
#include "../../StructureSynth/Parser/EisenParser.h"
#include "../../StructureSynth/Model/Rendering/OpenGLRenderer.h"
#include "../../StructureSynth/Model/Rendering/TemplateRenderer.h"
#include "../../StructureSynth/Model/Rendering/ObjRenderer.h"
#include "../../StructureSynth/Parser/Tokenizer.h"
#include "../../StructureSynth/Parser/Preprocessor.h"
#include "../../StructureSynth/Model/RuleSet.h"
#include "../../StructureSynth/Model/Builder.h"
#include "../../StructureSynth/JavaScriptSupport/JavaScriptParser.h"
#include "../../SyntopiaCore/GLEngine/Raytracer/RayTracer.h"
#include "../../SyntopiaCore/Math/Vector3.h"
#include "../../SyntopiaCore/Math/Random.h"
#include "../../SyntopiaCore/Math/Matrix4.h"

using namespace SyntopiaCore::Math;
using namespace SyntopiaCore::Logging;
using namespace StructureSynth::Model::Rendering;
using namespace SyntopiaCore::Exceptions;
using namespace StructureSynth::Parser;
using namespace StructureSynth::Model;
using namespace StructureSynth::JavaScriptSupport;
using namespace SyntopiaCore::GLEngine;

namespace StructureSynth {
	namespace GUI {

		namespace {
			int MaxRecentFiles = 5;

		
		}

		namespace {

			void createCommandHelpMenu(QMenu* menu, QWidget* textEdit) {

				QMenu *raytraceMenu = new QMenu("Raytracer Commands", 0);
				raytraceMenu->addAction("set raytracer::ambient-occlusion-samples 0 // turn off AO", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::samples 4 // for anti-alias and DOF", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::dof [0.4,0.1] // focal-plane distance, strength", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::shadows false", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::reflection 0.5", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::phong [0.6,0.6,0.3]", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::size [800x600]", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::size [0x800] // auto-calc proper width", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::size [800x0] // auto-calc proper height", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::max-threads 2", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::light [0,0,-29]", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::voxel-steps 30", textEdit , SLOT(insertText()));
				raytraceMenu->addAction("set raytracer::max-depth 5", textEdit , SLOT(insertText()));
				
				QMenu *modifierMenu = new QMenu("Rule Modifiers", 0);
				modifierMenu->addAction("weight", textEdit , SLOT(insertText()));
				modifierMenu->addAction("maxdepth", textEdit , SLOT(insertText()));
				modifierMenu->addAction("maxdepth > newRule", textEdit , SLOT(insertText()));

				QMenu *transformationMenu = new QMenu("Transformations", 0);
				transformationMenu->addAction("x 1 // translations", textEdit , SLOT(insertText()));
				transformationMenu->addAction("y 1", textEdit , SLOT(insertText()));
				transformationMenu->addAction("z 1", textEdit , SLOT(insertText()));
				transformationMenu->addAction("rx 45 // rotate angle around x-axis", textEdit , SLOT(insertText()));
				transformationMenu->addAction("ry 45", textEdit , SLOT(insertText()));
				transformationMenu->addAction("rz 45", textEdit , SLOT(insertText()));
				transformationMenu->addAction("hue 0.9", textEdit , SLOT(insertText()));
				transformationMenu->addAction("sat 0.9", textEdit , SLOT(insertText()));
				transformationMenu->addAction("brightness 0.9", textEdit , SLOT(insertText()));
				transformationMenu->addAction("color white // static color", textEdit , SLOT(insertText()));
				transformationMenu->addAction("color random // use color pool", textEdit , SLOT(insertText()));
				transformationMenu->addAction("blend red 0.5 // blend color with strength 0.5", textEdit , SLOT(insertText()));
				transformationMenu->addAction("alpha 0.9 // make transparent", textEdit , SLOT(insertText()));
				transformationMenu->addAction("matrix ", textEdit , SLOT(insertText()));
				transformationMenu->addAction("size 2 // uniform scaling", textEdit , SLOT(insertText()));
				transformationMenu->addAction("size 1 1 1.2 // non-uniform scaling", textEdit , SLOT(insertText()));
				transformationMenu->addAction("reflect", textEdit , SLOT(insertText()));
				transformationMenu->addAction("fx // mirror in yz-plane", textEdit , SLOT(insertText()));
				transformationMenu->addAction("fy", textEdit , SLOT(insertText()));
				transformationMenu->addAction("fz", textEdit , SLOT(insertText()));

				QMenu *setMenu = new QMenu("Set Commands", 0);
				setMenu->addAction("set maxdepth 100", textEdit , SLOT(insertText()));
				setMenu->addAction("set maxobjects 100", textEdit , SLOT(insertText()));
				setMenu->addAction("set maxsize 10 // maximum object size", textEdit , SLOT(insertText()));
				setMenu->addAction("set minsize 0.1 // minimum object size", textEdit , SLOT(insertText()));
				setMenu->addAction("set background #fff", textEdit , SLOT(insertText()));
				setMenu->addAction("set seed 45", textEdit , SLOT(insertText()));

				QMenu *colorMenu = new QMenu("Set Colorpool Commands", 0);
				colorMenu->addAction("set colorpool randomrgb", textEdit , SLOT(insertText()));
				colorMenu->addAction("set colorpool randomhue", textEdit , SLOT(insertText()));
				colorMenu->addAction("set colorpool list:orange,yellow,blue // sample from list", textEdit , SLOT(insertText()));
				colorMenu->addAction("set colorpool image:test.png // sample from image", textEdit , SLOT(insertText()));

				QMenu *set2Menu = new QMenu("Exotic Set Commands", 0);
				set2Menu->addAction("set seed initial // reset random seed", textEdit , SLOT(insertText()));
				set2Menu->addAction("set recursion depth // traverse depth-first", textEdit , SLOT(insertText()));
				set2Menu->addAction("set rng old // old random generator", textEdit , SLOT(insertText()));
				set2Menu->addAction("set syncrandom true", textEdit , SLOT(insertText()));				

				QMenu *setCMenu = new QMenu("Camera Commands", 0);
				setCMenu->addAction("set scale 0.5", textEdit , SLOT(insertText()));
				setCMenu->addAction("set pivot [0 0 0]", textEdit , SLOT(insertText()));
				setCMenu->addAction("set translation [0 0 -20]", textEdit , SLOT(insertText()));
				setCMenu->addAction("set rotation  [1 0 0 0 1 0 0 0 1]", textEdit , SLOT(insertText()));
				setCMenu->addAction("set perspective-angle 70", textEdit , SLOT(insertText()));

				QMenu *pMenu = new QMenu("Preprocessor Commands", 0);
				pMenu->addAction("#define myAngle 45", textEdit , SLOT(insertText()));
				pMenu->addAction("#define angle 20 (float:0-90) // create slider with default value 20", textEdit , SLOT(insertText()));
				pMenu->addAction("#define iterations 20 (int:0-23) // create slider with default value 20", textEdit , SLOT(insertText()));

				QMenu *p2Menu = new QMenu("JavaScript Commands", 0);
				p2Menu->addAction("Builder.load(\"system.es\");", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.define(\"_rotation\", 34);", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.prepend(\"sometext\");", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.append(\"sometext\");", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.build();", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.renderToFile(\"out.png\", overwrite);", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.raytraceToFile(\"out.png\", overwrite);", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.templateRenderToFile(\"Sunflow-Colored.rendertemplate\", \"sunflow.sc\", overwrite);", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.execute('\"%JAVAPATH%/java\"', '-Xmx1G -server -jar \"%SUNFLOW%/sunflow.jar\" sunflow.sc -nogui -o sunflow.png\", true);", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.reset();", textEdit , SLOT(insertText()));
				p2Menu->addAction("Builder.setSize(640,0);", textEdit , SLOT(insertText()));
				
				menu->insertMenu(menu->actions()[0], modifierMenu);
				menu->insertMenu(menu->actions()[1], transformationMenu);
				menu->insertMenu(menu->actions()[2], setMenu);
				menu->insertMenu(menu->actions()[3], set2Menu);
				menu->insertMenu(menu->actions()[4], colorMenu);
				menu->insertMenu(menu->actions()[5], raytraceMenu);
				menu->insertMenu(menu->actions()[6], setCMenu);
				menu->insertMenu(menu->actions()[7], pMenu);
				menu->insertMenu(menu->actions()[8], p2Menu);
				menu->insertSeparator(menu->actions()[9]);

			}
		}
		void TextEdit::contextMenuEvent(QContextMenuEvent *event)
		{	
			QMenu *menu = createStandardContextMenu();

			createCommandHelpMenu(menu, this);

			menu->exec(event->globalPos());
			delete menu;
		}

		void TextEdit::insertText() {
			QString text = ((QAction*)sender())->text();
			insertPlainText(text.section("//",0,0)); // strip comments
		}


		class EisenScriptHighlighter : public QSyntaxHighlighter {
		public:

			EisenScriptHighlighter(QTextEdit* e) : QSyntaxHighlighter(e) {
				keywordFormat.setFontWeight(QFont::Bold);
				keywordFormat.setForeground(Qt::darkMagenta);
				bracketFormat.setFontWeight(QFont::Bold);
				bracketFormat.setForeground(Qt::blue);
				primitiveFormat.setFontWeight(QFont::Bold);
				primitiveFormat.setForeground(Qt::darkYellow);
				commentFormat.setForeground(Qt::darkGreen);
				warningFormat.setBackground(QBrush(Qt::yellow));
				preprocessorFormat.setForeground(QBrush(Qt::blue));
				preprocessorFormat.setFontWeight(QFont::Bold);

				expression = QRegExp("(set|blend|recursion|maxsize|minsize|translation|pivot|rotation|scale|maxobjects|background|color|rule|a|alpha|matrix|h|hue|sat|b|brightness|v|x|y|z|rx|ry|rz|s|fx|fy|fz|maxdepth|weight|md|w)");
				primitives = QRegExp("(triangle\\[.*\\]|sphere(::\\w+)?|box(::\\w+)?|dot(::\\w+)?|line(::\\w+)?|grid(::\\w+)?)");
				randomNumber = QRegExp("(random\\[[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+\\])"); // random[-2.3,3.4]

				expression.setCaseSensitivity(Qt::CaseInsensitive);
				primitives.setCaseSensitivity(Qt::CaseInsensitive);
				randomNumber.setCaseSensitivity(Qt::CaseInsensitive);
			};

			void highlightBlock(const QString &text)
			{

				if (currentBlockState() == 2) {
					setFormat(0, text.length(), warningFormat);
					setCurrentBlockState(-1);
					return;
				}


				if (previousBlockState() != 1 && currentBlockState() == 1) {
					// This line was previously a multi-line start 
					if (!text.contains("*/")) setCurrentBlockState(0);
				}

				if (previousBlockState() == 1) {
					// Part of multi-line comment. Skip the rest...
					if (!text.contains("*/")) {
						setFormat(0, text.length(), commentFormat);
						setCurrentBlockState(1);
						return;
					}
				}

				// Line parsing
				QString current;
				int startMatch = 0;
				for (int i = 0; i < text.length(); i++) {
					if ((i > 0) && text.at(i) == '*' && text.at(i-1) == '/') {
						// Multi-line comment begins
						setFormat(i-1, text.length()-i+1, commentFormat);
						setCurrentBlockState(1);
						return;
					}

					if ((i > 0) && text.at(i) == '/' && text.at(i-1) == '*') {
						// Multi-line comment ends
						setFormat(0, i, commentFormat);
						if (currentBlockState() != 0) {
							setCurrentBlockState(0);
						}
						continue;
					}

					if (text.at(0) == '#') {
						// Preprocessor format
						setFormat(0, text.length(), preprocessorFormat);
						continue;
					}

					if ((i > 0) && (i < text.length()-2) && text.at(i) == '/' && text.at(i-1) == '/') {
						// Single-line comments
						setFormat(i-1, text.length()-i+1, commentFormat);
						break;
					}

					bool delimiter = (text.at(i) == '{' || text.at(i) == '\t' || text.at(i) == '}' || text.at(i) == ' '  || (text.at(i) == '\r') || (text.at(i) == '\n'));
					bool lastChar = (i==text.length()-1);
					if (delimiter || lastChar) {
						if (lastChar && !delimiter) current += text.at(i);
						int adder = (i==text.length()-1 ? 1 : 0);
						if (expression.exactMatch(current)) setFormat(startMatch, i-startMatch+adder, keywordFormat);
						if (primitives.exactMatch(current)) setFormat(startMatch, i-startMatch+adder, primitiveFormat);
						if (randomNumber.exactMatch(current)) setFormat(startMatch, i-startMatch+adder, preprocessorFormat);
						if (text.at(i) == '{' || text.at(i) == '}') setFormat(i, 1, bracketFormat);
						startMatch = i;
						current = "";
					} else {
						current += text.at(i);
					}
				}

			}; 
		private:
			QTextCharFormat keywordFormat;
			QTextCharFormat bracketFormat;
			QTextCharFormat primitiveFormat;
			QTextCharFormat commentFormat;
			QTextCharFormat warningFormat;
			QTextCharFormat preprocessorFormat;

			QRegExp expression;
			QRegExp primitives;
			QRegExp randomNumber;
		};


		MainWindow::MainWindow()
		{
			init();
			loadFile(QDir(getExamplesDir()).absoluteFilePath("Default.es"));
			tabChanged(0); // to update title.
		}

		MainWindow::MainWindow(const QString &fileName)
		{
			QDir::setCurrent(QCoreApplication::applicationDirPath ()); // Otherwise we cannot find examples + templates
			init();
			loadFile(fileName);
			tabChanged(0); // to update title.
		}

		void MainWindow::closeEvent(QCloseEvent *ev)
		{

			bool modification = false;
			for (int i = 0; i < tabInfo.size(); i++) {
				if (tabInfo[i].unsaved) modification = true;
			}

			if (modification) {
				int i = QMessageBox::warning(this, "Unsaved changed", "There are tabs with unsaved changes.\r\nContinue and loose changes?", QMessageBox::Ok, QMessageBox::Cancel);
				if (i == QMessageBox::Ok) {
					// OK
					ev->accept();
					return;
				} else {
					// Cancel
					ev->ignore();
					return;
				}
			}
			ev->accept();

			writeSettings();
		}

		void MainWindow::newFile()
		{
			insertTabPage("");
		}

		void MainWindow::open()
		{
			QString filter = "EisenScript (*.es);;All Files (*.*)";
			QString fileName = QFileDialog::getOpenFileName(this, QString(), QString(), filter);
			if (!fileName.isEmpty()) {
				loadFile(fileName);
			}
		}

		void MainWindow::keyReleaseEvent(QKeyEvent* ev) {
			if (ev->key() == Qt::Key_Escape) {
				toggleFullScreen();
			} else if (ev->key() == Qt::Key_F5 && fullScreenEnabled) {
				render();
			} else if (ev->key() == Qt::Key_F6 && fullScreenEnabled) {
				templateExport();
			} else {
				ev->ignore();
			}
		};


		bool MainWindow::save()
		{
			int index = tabBar->currentIndex();
			if (index == -1) { WARNING("No open tab"); return false; } 
			TabInfo t = tabInfo[index];

			if (t.hasBeenSavedOnce) {
				return saveFile(t.filename);
			} else {
				return saveAs();
			}
		}

		bool MainWindow::saveAs()
		{
			int index = tabBar->currentIndex();
			if (index == -1) { WARNING("No open tab"); return false; } 

			TabInfo t = tabInfo[index];

			QString filter = "EisenScript (*.es);;All Files (*.*)";

			QString fileName = QFileDialog::getSaveFileName(this, tr("Save As"), t.filename, filter);
			if (fileName.isEmpty())
				return false;

			return saveFile(fileName);
		}

		void MainWindow::about()
		{

			QFile file(getMiscDir() + QDir::separator() + "about.html");
			if (!file.open(QFile::ReadOnly | QFile::Text)) {
				WARNING("Could not open about.html...");
				return;
			}

			QTextStream in(&file);
			QString text = in.readAll();

			text.replace("$VERSION$", version.toLongString());

			QMessageBox mb(this);
			mb.setText(text);
			mb.setWindowTitle("About Structure Synth");
			mb.setIconPixmap(getMiscDir() + QDir::separator() + "icon.jpg");
			mb.setMinimumWidth(800);
			mb.exec();

		}

		void MainWindow::documentWasModified()
		{
			tabInfo[tabBar->currentIndex()].unsaved = true;
			tabChanged(tabBar->currentIndex());
		}

		void MainWindow::init()
		{
			setAcceptDrops(true);

			hasBeenResized = true;

			oldDirtyPosition = -1;
			setFocusPolicy(Qt::StrongFocus);

			version = SyntopiaCore::Misc::Version(1, 5, 0, -1, " (\"Hinxton\")");
			setAttribute(Qt::WA_DeleteOnClose);

			QSplitter*	splitter = new QSplitter(this);
			splitter->setObjectName(QString::fromUtf8("splitter"));
			splitter->setOrientation(Qt::Horizontal);

			stackedTextEdits = new QStackedWidget(splitter);

			engine = new SyntopiaCore::GLEngine::EngineWidget(this,splitter);

			tabBar = new QTabBar(this);

#if QT_VERSION >= 0x040500
			tabBar->setTabsClosable(true);
			connect(tabBar, SIGNAL(tabCloseRequested(int)), this, SLOT(closeTab(int)));
#endif 

			QFrame* f = new QFrame(this);
			frameMainWindow = new QVBoxLayout();
			frameMainWindow->setSpacing(0);
			frameMainWindow->setMargin(4);
			f->setLayout(frameMainWindow);
			f->layout()->addWidget(tabBar);
			f->layout()->addWidget(splitter);
			setCentralWidget(f);

			QList<int> l; l.append(100); l.append(400);
			splitter->setSizes(l);

			createActions();
			
			QDir d(getExamplesDir());

			// Log widget (in dockable window)
			dockLog = new QDockWidget(this);
			dockLog->setWindowTitle("Log");
			dockLog->setObjectName(QString::fromUtf8("dockWidget"));
			dockLog->setAllowedAreas(Qt::BottomDockWidgetArea);
			QWidget* dockLogContents = new QWidget(dockLog);
			dockLogContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
			QVBoxLayout* vboxLayout1 = new QVBoxLayout(dockLogContents);
			vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
			vboxLayout1->setContentsMargins(0, 0, 0, 0);

			ListWidgetLogger* logger = new ListWidgetLogger(dockLog);
			vboxLayout1->addWidget(logger->getListWidget());
			dockLog->setWidget(dockLogContents);
			addDockWidget(static_cast<Qt::DockWidgetArea>(8), dockLog);

			// Variable editor (in dockable window)
			editorDockWidget = new QDockWidget(this);
			editorDockWidget->setMinimumWidth(250);
			editorDockWidget->setWindowTitle("Variables");
			editorDockWidget->setObjectName(QString::fromUtf8("editorDockWidget"));
			editorDockWidget->setAllowedAreas(Qt::RightDockWidgetArea);
			QWidget* editorLogContents = new QWidget(dockLog);
			editorLogContents->setObjectName(QString::fromUtf8("editorLogContents"));
			QVBoxLayout* vboxLayout2 = new QVBoxLayout(editorLogContents);
			vboxLayout2->setObjectName(QString::fromUtf8("vboxLayout2"));
			vboxLayout2->setContentsMargins(0, 0, 0, 0);

			variableEditor = new VariableEditor(editorDockWidget);
			variableEditor->setMinimumWidth(250);
			vboxLayout2->addWidget(variableEditor);
			editorDockWidget->setWidget(editorLogContents);
			addDockWidget(Qt::RightDockWidgetArea, editorDockWidget);

			editorDockWidget->setHidden(true);
			setMouseTracking(true);
			probeDepth = false;

			INFO(QString("Welcome to Structure Synth version %1. A Syntopia Project.").arg(version.toLongString()));
			INFO("");
			INFO("Zoom by pressing both mouse buttons, holding SHIFT+left mouse button, or using scroll wheel. Translate using right mouse button. Hold 'ALT' for fast rotate (quick draw mode).");
			INFO("Press 'Reset View' if you get lost in the 3D view.");
			INFO("Use the context menu in the text edit area to get a list of common commands.");
			INFO("");
			INFO("Please report bugs and feature requests at the SourceForge forums (weblink at the Help Menu). Enjoy.");
			//WARNING("This is an experimental SVN checkout build. For stability use the package releases.");

			fullScreenEnabled = false;
			createOpenGLContextMenu();

			connect(this->tabBar, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));

			readSettings();

			//m->addMenu(createPopupMenu());

			createToolBars();
			createStatusBar();
			createMenus();
			
		}



		void MainWindow::createOpenGLContextMenu() {
			openGLContextMenu = new QMenu();			
			openGLContextMenu->addAction(insertCameraSettingsAction);

			probeDepthAction  = new QAction(tr("Show 3D Object Information"), this);
			probeDepthAction->setCheckable(true);
			connect(probeDepthAction, SIGNAL(triggered()), this, SLOT(toggleProbeDepth()));
			openGLContextMenu->addAction(probeDepthAction);

			openGLContextMenu->addAction(fullScreenAction);
			openGLContextMenu->addAction(screenshotAction);
			openGLContextMenu->addAction(resetViewAction);
			engine->setContextMenu(openGLContextMenu);
		}


		void MainWindow::toggleFullScreen() {
			if (fullScreenEnabled) {
				frameMainWindow->setMargin(4);
				showNormal();
				fullScreenEnabled = false;
				fullScreenAction->setChecked(false);
				stackedTextEdits->show();
				dockLog->show();
				menuBar()->show();
				statusBar()->show();
				fileToolBar->show();
				editToolBar->show();
				renderToolBar->show();
				tabBar->show();
				randomToolBar->show();
			} else {
				frameMainWindow->setMargin(0);
				fullScreenAction->setChecked(true);
				fullScreenEnabled = true;

				tabBar->hide();
				stackedTextEdits->hide();
				dockLog->hide();
				menuBar()->hide();
				statusBar()->hide();
				fileToolBar->hide();
				editToolBar->hide();
				renderToolBar->hide();
				randomToolBar->hide();
				showFullScreen();
			}
		}


		void MainWindow::toggleProbeDepth() {
			getEngine()->toggleShowDepth();
		}

		void MainWindow::createActions()
		{
			fullScreenAction = new QAction(tr("F&ullscreen"), this);
			fullScreenAction->setShortcut(tr("Ctrl+F"));
			fullScreenAction->setCheckable(true);
			connect(fullScreenAction, SIGNAL(triggered()), this, SLOT(toggleFullScreen()));

			fastRotateAction = new QAction(tr("Fast Rotate"), this);
			fastRotateAction->setCheckable(true);
			connect(fastRotateAction, SIGNAL(triggered()), this, SLOT(fastRotateChanged()));

			showCoordinateSystemAction = new QAction(tr("Show Coordinate System"), this);
			showCoordinateSystemAction->setCheckable(true);
			connect(showCoordinateSystemAction, SIGNAL(triggered()), this, SLOT(showCoordinateSystemChanged()));


			insertCameraSettingsAction  = new QAction(tr("&Copy Camera Settings to EisenScript Window"), this);
			connect(insertCameraSettingsAction, SIGNAL(triggered()), this, SLOT(insertCameraSettings()));

			screenshotAction = new QAction(tr("&Save as Bitmap..."), this);
			connect(screenshotAction, SIGNAL(triggered()), this, SLOT(makeScreenshot()));

			raytraceFinalAction = new QAction(tr("&Raytrace (Final)"), this);
			connect(raytraceFinalAction, SIGNAL(triggered()), this, SLOT(raytrace()));

			raytraceProgressiveAction = new QAction(tr("&Raytrace (in Window)"), this);
			connect(raytraceProgressiveAction, SIGNAL(triggered()), this, SLOT(raytraceProgressive()));

			newAction = new QAction(QIcon(":/images/new.png"), tr("&New"), this);
			newAction->setShortcut(tr("Ctrl+N"));
			newAction->setStatusTip(tr("Create a new file"));
			connect(newAction, SIGNAL(triggered()), this, SLOT(newFile()));

			openAction = new QAction(QIcon(":/images/open.png"), tr("&Open..."), this);
			openAction->setShortcut(tr("Ctrl+O"));
			openAction->setStatusTip(tr("Open an existing file"));
			connect(openAction, SIGNAL(triggered()), this, SLOT(open()));

			saveAction = new QAction(QIcon(":/images/save.png"), tr("&Save"), this);
			saveAction->setShortcut(tr("Ctrl+S"));
			saveAction->setStatusTip(tr("Save the script to disk"));
			connect(saveAction, SIGNAL(triggered()), this, SLOT(save()));

			saveAsAction = new QAction(QIcon(":/images/filesaveas.png"), tr("Save &As..."), this);
			saveAsAction->setStatusTip(tr("Save the script under a new name"));
			connect(saveAsAction, SIGNAL(triggered()), this, SLOT(saveAs()));

			closeAction = new QAction(QIcon(":/images/fileclose.png"), tr("&Close Tab"), this);
			closeAction->setShortcut(tr("Ctrl+W"));
			closeAction->setStatusTip(tr("Close this tab"));
			connect(closeAction, SIGNAL(triggered()), this, SLOT(closeTab()));

			exitAction = new QAction(QIcon(":/images/exit.png"), tr("E&xit Application"), this);
			exitAction->setShortcut(tr("Ctrl+Q"));
			exitAction->setStatusTip(tr("Exit the application"));
			connect(exitAction, SIGNAL(triggered()), this, SLOT(close()));

			cutAction = new QAction(QIcon(":/images/cut.png"), tr("Cu&t"), this);
			cutAction->setShortcut(tr("Ctrl+X"));
			cutAction->setStatusTip(tr("Cut the current selection's contents to the "
				"clipboard"));
			connect(cutAction, SIGNAL(triggered()), this, SLOT(cut()));

			copyAction = new QAction(QIcon(":/images/copy.png"), tr("&Copy"), this);
			copyAction->setShortcut(tr("Ctrl+C"));
			copyAction->setStatusTip(tr("Copy the current selection's contents to the "
				"clipboard"));
			connect(copyAction, SIGNAL(triggered()), this, SLOT(copy()));

			pasteAction = new QAction(QIcon(":/images/paste.png"), tr("&Paste"), this);
			pasteAction->setShortcut(tr("Ctrl+V"));
			pasteAction->setStatusTip(tr("Paste the clipboard's contents into the current "
				"selection"));
			connect(pasteAction, SIGNAL(triggered()), this, SLOT(paste()));

			renderAction = new QAction(QIcon(":/images/render.png"), tr("&Build System"), this);
			renderAction->setShortcut(tr("F5"));
			renderAction->setStatusTip(tr("Render the current ruleset"));
			connect(renderAction, SIGNAL(triggered()), this, SLOT(render()));

			exportAction = new QAction(tr("&Template Export..."), this);
			exportAction->setShortcut(tr("F6"));
			exportAction->setStatusTip(tr("Export the structure using a template."));
			connect(exportAction, SIGNAL(triggered()), this, SLOT(templateExport()));

			resetViewAction = new QAction("Reset View", this);
			resetViewAction->setStatusTip(tr("Resets the viewport"));
			connect(resetViewAction, SIGNAL(triggered()), this, SLOT(resetView()));

			aboutAction = new QAction(QIcon(":/images/documentinfo.png"), tr("&About"), this);
			aboutAction->setStatusTip(tr("Show the About box"));
			connect(aboutAction, SIGNAL(triggered()), this, SLOT(about()));

			sfHomeAction = new QAction(QIcon(":/images/agt_internet.png"), tr("&Project Homepage (web link)"), this);
			sfHomeAction->setStatusTip(tr("Open the SourceForge project page in a browser."));
			connect(sfHomeAction, SIGNAL(triggered()), this, SLOT(launchSfHome()));

			referenceAction = new QAction(QIcon(":/images/agt_internet.png"), tr("&Structure Synth Reference (web link)"), this);
			referenceAction->setStatusTip(tr("Open a Structure Synth reference web page in a browser."));
			connect(referenceAction, SIGNAL(triggered()), this, SLOT(launchReferenceHome()));

			galleryAction = new QAction(QIcon(":/images/agt_internet.png"), tr("&Flickr Structure Synth Group (web link)"), this);
			galleryAction->setStatusTip(tr("Opens the main Flickr group for Structure Synth creations."));
			connect(galleryAction, SIGNAL(triggered()), this, SLOT(launchGallery()));

			for (int i = 0; i < MaxRecentFiles; ++i) {
				QAction* a = new QAction(this);
				a->setVisible(false);
				connect(a, SIGNAL(triggered()),	this, SLOT(openFile()));
				recentFileActions.append(a);				
			}

			qApp->setWindowIcon(QIcon(":/images/structuresynth.png"));
		}

		void MainWindow::createMenus()
		{
			// -- File Menu --
			fileMenu = menuBar()->addMenu(tr("&File"));
			fileMenu->addAction(newAction);
			fileMenu->addAction(openAction);
			fileMenu->addAction(saveAction);
			fileMenu->addAction(saveAsAction);
			recentFileSeparator = fileMenu->addSeparator();
			for (int i = 0; i < MaxRecentFiles; ++i) fileMenu->addAction(recentFileActions[i]);
			fileMenu->addSeparator();
			fileMenu->addAction(closeAction);
			fileMenu->addAction(exitAction);

			// -- Edit Menu --
			editMenu = menuBar()->addMenu(tr("&Edit"));
			editMenu->addAction(cutAction);
			editMenu->addAction(copyAction);
			editMenu->addAction(pasteAction);
			editMenu->addSeparator();
			editMenu->addAction(insertCameraSettingsAction);
			QMenu* m = editMenu->addMenu("Insert Command");
			createCommandHelpMenu(m, this);


			// -- Render Menu --
			renderMenu = menuBar()->addMenu(tr("&Render"));
			renderMenu->addAction(renderAction);
			renderMenu->addSeparator();
			renderMenu->addAction(raytraceProgressiveAction);
			renderMenu->addAction(raytraceFinalAction);
			renderMenu->addSeparator();
			renderMenu->addAction(fullScreenAction);
			renderMenu->addAction(resetViewAction);
			renderMenu->addSeparator();
			
			renderMenu->addAction(fastRotateAction);
			renderMenu->addAction(showCoordinateSystemAction);
			renderMenu->addAction(probeDepthAction);
			

			// -- Export --
			renderMenu = menuBar()->addMenu(tr("&Export"));
			renderMenu->addAction(exportAction);
			renderMenu->addAction("Obj Export...", this, SLOT(exportToObj()));
			renderMenu->addAction(screenshotAction);

		
			//menuBar()->addSeparator();

			// -- Examples Menu --
			QStringList filters;
			QMenu* examplesMenu = menuBar()->addMenu(tr("&Examples"));
			// Scan examples dir...
			QDir d(getExamplesDir());
			filters.clear();
			filters << "*.es";
			d.setNameFilters(filters);
			if (!d.exists()) {
				QAction* a = new QAction("Unable to locate: "+d.absolutePath(), this);
				a->setEnabled(false);
				examplesMenu->addAction(a);
			} else {
				// we will recurse the dirs...
				QStack<QString> pathStack;
				pathStack.append(QDir(getExamplesDir()).absolutePath());

				QMap< QString , QMenu* > menuMap;
				while (!pathStack.isEmpty()) {

					QMenu* currentMenu = examplesMenu;
					QString path = pathStack.pop();
					if (menuMap.contains(path)) currentMenu = menuMap[path];
					QDir dir(path);

					QStringList sl = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
					for (int i = 0; i < sl.size(); i++) {
						QMenu* menu = new QMenu(sl[i]);
						QString absPath = QDir(path + QDir::separator() +  sl[i]).absolutePath();
						menuMap[absPath] = menu;
						currentMenu->addMenu(menu);
						menu->setIcon(QIcon(":/images/folder.png"));
						pathStack.push(absPath);
					}

					dir.setNameFilters(filters);

					sl = dir.entryList();
					for (int i = 0; i < sl.size(); i++) {
						QAction* a = new QAction(sl[i], this);
						a->setIcon(QIcon(":/images/mail_new.png"));


						QString absPath = QDir(path ).absoluteFilePath(sl[i]);

						a->setData(absPath);
						connect(a, SIGNAL(triggered()), this, SLOT(openFile()));
						currentMenu->addAction(a);
					}
				}


			}

			QMenu* mc = createPopupMenu();
			mc->setTitle("Windows");
			menuBar()->addMenu(mc);


			helpMenu = menuBar()->addMenu(tr("&Help"));
			helpMenu->addAction(aboutAction);
			helpMenu->addSeparator();
			helpMenu->addAction(sfHomeAction);
			helpMenu->addAction(referenceAction);
			helpMenu->addAction(galleryAction);
		}

		void MainWindow::createToolBars()
		{
			fileToolBar = addToolBar(tr("File Toolbar"));
			fileToolBar->addAction(newAction);
			fileToolBar->addAction(openAction);
			fileToolBar->addAction(saveAction);

			editToolBar = addToolBar(tr("Edit Toolbar"));
			editToolBar->addAction(cutAction);
			editToolBar->addAction(copyAction);
			editToolBar->addAction(pasteAction);

			randomToolBar = addToolBar(tr("Random Toolbar"));

			QLabel* randomSeed = new QLabel("Seed:"); 
			randomToolBar->addWidget(randomSeed);
			seedSpinBox = new QSpinBox();
			seedSpinBox->setRange(1,32768);
			seedSpinBox->setValue(1);
			randomToolBar->addWidget(seedSpinBox);
			autoIncrementCheckbox = new QCheckBox("Auto Increment", randomToolBar);
			randomToolBar->addWidget(autoIncrementCheckbox);
			autoIncrementCheckbox->setChecked(true);

			renderToolBar = addToolBar(tr("Render Toolbar"));
			renderToolBar->addAction(renderAction);
			renderToolBar->addWidget(new QLabel("Build    ", this));
			QPushButton* pb = new QPushButton(this);
			pb->setText("Reset View");
			renderToolBar->addWidget(pb);
			connect(pb, SIGNAL(clicked()), this, SLOT(resetView()));
			
			progressBox = new ProgressBox(this);
			renderToolBar->addWidget(progressBox);
			connect(progressBox, SIGNAL(startPressed()), this, SLOT(raytraceProgressive()));
			progressBox->setValue(0);
			connect(seedSpinBox, SIGNAL(valueChanged(int)), this, SLOT(seedChanged()));
		}

		void MainWindow::fastRotateChanged() {
			engine->setFastRotate(fastRotateAction->isChecked());
		}

		void MainWindow::showCoordinateSystemChanged() {
			engine->setShowCoordinateSystem(showCoordinateSystemAction->isChecked());
			if (showCoordinateSystemAction->isChecked()) {
				INFO("X,Y,Z axis is colored red,green, and blue respectively. Axis length is 10 units.");
			}
			engine->requireRedraw();
		}

		void MainWindow::disableAllExcept(QWidget* w) {
			disabledWidgets.clear();
			disabledWidgets = findChildren<QWidget *>("");
			while (w) { disabledWidgets.removeAll(w); w=w->parentWidget(); }
			foreach (QWidget* w, disabledWidgets) w->setEnabled(false);
			qApp->processEvents();
		}

		void MainWindow::enableAll() {
			foreach (QWidget* w, disabledWidgets) w->setEnabled(true);
		}
		
		void MainWindow::raytraceProgressive() {
			disableAllExcept(progressBox);
			RayTracer rt(engine, progressBox,true);
			QImage im = rt.calculateImage(engine->width(),engine->height());
			engine->setImage(im);
			enableAll();
		}




		void MainWindow::seedChanged() {
			autoIncrementCheckbox->setChecked(false);
		}

		void MainWindow::createStatusBar()
		{
			statusBar()->showMessage(tr("Ready"));
		}

		void MainWindow::readSettings()
		{
			QSettings settings("Syntopia Software", "Structure Synth");
			QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
			QSize size = settings.value("size", QSize(1024, 800)).toSize();
			move(pos);
			resize(size);
		}

		void MainWindow::writeSettings()
		{
			QSettings settings;
			settings.setValue("pos", pos());
			settings.setValue("size", size());
		}


		void MainWindow::openFile()
		{
			QAction *action = qobject_cast<QAction *>(sender());
			if (action) {
				loadFile(action->data().toString());
			} else {
				WARNING("No data!");
			}
		}

		void MainWindow::loadFile(const QString &fileName)
		{
			insertTabPage(fileName);
		}

		bool MainWindow::saveFile(const QString &fileName)
		{
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return false; } 

			QFile file(fileName);
			if (!file.open(QFile::WriteOnly | QFile::Text)) {
				QMessageBox::warning(this, tr("Structure Synth"),
					tr("Cannot write file %1:\n%2.")
					.arg(fileName)
					.arg(file.errorString()));
				return false;
			}

			QTextStream out(&file);
			QApplication::setOverrideCursor(Qt::WaitCursor);
			out << getTextEdit()->toPlainText();
			QApplication::restoreOverrideCursor();

			tabInfo[tabBar->currentIndex()].hasBeenSavedOnce = true;
			tabInfo[tabBar->currentIndex()].unsaved = false;
			tabInfo[tabBar->currentIndex()].filename = fileName;
			tabChanged(tabBar->currentIndex()); // to update displayed name;

			statusBar()->showMessage(tr("File saved"), 2000);
			setRecentFile(fileName);

			return true;
		}



		QString MainWindow::strippedName(const QString &fullFileName)
		{
			return QFileInfo(fullFileName).fileName();
		}


		void MainWindow::updateRandom() {

			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 

			seedSpinBox->blockSignals(true);
			setSeed((getSeed()+1) % 32768);
			seedSpinBox->blockSignals(false);
			INFO(QString("Auto-incremented random seed: %1").arg(getSeed()));

			// Should we try something like below?
			if (tabInfo[tabBar->currentIndex()].unsaved) {
				// Current tab is unsaved, we will not change the random seed.
			} else {
				// We will auto-increment random seed.
			}

			RandomStreams::SetSeed(getSeed());
		}

		void MainWindow::render() {
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 

			if (getTextEdit()->toPlainText().startsWith("#javascript", Qt::CaseInsensitive)) {
				// This is javascript...
				QString text = getTextEdit()->toPlainText();
				text = text.remove("#javascript", Qt::CaseInsensitive);

				TabInfo t = tabInfo[tabBar->currentIndex()];
				QString dir;
				if (t.hasBeenSavedOnce) {
					dir = QFileInfo(t.filename).absolutePath();
				} else {
					dir = ""; // we will use the default dir.
				}

				parseJavaScript(text,dir);
				return;
			}

			setEnabled(false);
			if (autoIncrementCheckbox->isChecked()) updateRandom();
			RandomStreams::SetSeed(getSeed());
			INFO(QString("Random seed: %1").arg(getSeed()));

			engine->setDisabled(true);

			try {

				Rendering::OpenGLRenderer renderTarget(engine);
				renderTarget.begin(); // we clear before parsing...

				Preprocessor pp;
				QString out = pp.Process(getTextEdit()->toPlainText(), getSeed());
				bool showGUI = false;
				out = variableEditor->updateFromPreprocessor(&pp, out, &showGUI);
				editorDockWidget->setHidden(!showGUI);

				Tokenizer tokenizer(out);
				EisenParser e(&tokenizer);
				INFO("Started Eisenstein engine...");
				RuleSet* rs = e.parseRuleset();

				rs->resolveNames();

				rs->dumpInfo();

				Builder b(&renderTarget, rs, true);
				b.build();
				renderTarget.end();

				//INFO("Done...");

				if (b.seedChanged()) {
					setSeed(b.getNewSeed());
					INFO(QString("Builder changed seed to: %1").arg(b.getNewSeed()));
				} 

				if (oldDirtyPosition > 0) {
					getTextEdit()->document()->markContentsDirty(oldDirtyPosition,1);
					oldDirtyPosition = -1;
				}

				engine->setRaytracerCommands(b.getRaytracerCommands());
				INFO(QString("Setting %1 raytracer commands.").arg(b.getRaytracerCommands().count()));

				delete(rs); 
				rs = 0;

			} catch (ParseError& pe) {
				WARNING(pe.getMessage());
				int pos = pe.getPosition();
				INFO(QString("Found at character %1.").arg(pos));	
				getTextEdit()->document()->findBlock(pos).setUserState(2);
				getTextEdit()->document()->markContentsDirty(pos,1);
				oldDirtyPosition = pos;
				engine->clearWorld();
				engine->requireRedraw();
			} catch (Exception& er) {
				WARNING(er.getMessage());
				engine->clearWorld();
				engine->requireRedraw();
			} 
			setEnabled(true);

			engine->setDisabled(false);

		}

		namespace {
			// Returns the first valid directory.
			QString findDirectory(QStringList guesses) {
				QStringList invalid;
				for (int i = 0; i < guesses.size(); i++) {
					if (QFile::exists(guesses[i])) return guesses[i];
					invalid.append(QFileInfo(guesses[i]).absoluteFilePath());
				}

				// not found.
				WARNING("Could not locate directory in: " + invalid.join(",") + ".");
				return "[not found]";
			}
		}

		// Mac needs to step two directies up, when debugging in XCode...
		QString MainWindow::getExamplesDir() {
			QStringList examplesDir;
			examplesDir << "Examples" << "../../Examples";
			return findDirectory(examplesDir);
		}

		QString MainWindow::getMiscDir() {
			QStringList miscDir;
			miscDir << "Misc" << "../../Misc";
			return findDirectory(miscDir);
		}

		QString MainWindow::getTemplateDir() {
			return getMiscDir();
		}

		void MainWindow::resetView() {
			engine->reset();
		}

		QTextEdit* MainWindow::getTextEdit() {
			return (stackedTextEdits->currentWidget() ? (QTextEdit*)stackedTextEdits->currentWidget() : 0);
		}

		void MainWindow::cursorPositionChanged() {
			if (!this->getTextEdit()) return;
			int pos = this->getTextEdit()->textCursor().position();
			int blockNumber = this->getTextEdit()->textCursor().blockNumber();
			statusBar()->showMessage(QString("Position: %1, Line: %2").arg(pos).arg(blockNumber+1), 5000);
		}

		void MainWindow::insertTabPage(QString filename) {
			QTextEdit* textEdit = new TextEdit();
			connect(textEdit, SIGNAL(cursorPositionChanged()), this, SLOT(cursorPositionChanged()));

			textEdit->setLineWrapMode(QTextEdit::NoWrap);
			textEdit->setTabStopWidth(20);
			new EisenScriptHighlighter(textEdit);

			QString s = QString("// Write EisenScript code here...\r\n");
			textEdit->setText(s);

			bool loadingSucceded = false;
			if (!filename.isEmpty()) {
				INFO(QString("Loading file: %1").arg(filename));
				QFile file(filename);
				if (!file.open(QFile::ReadOnly | QFile::Text)) {
					textEdit->setPlainText(QString("Cannot read file %1:\n%2.").arg(filename).arg(file.errorString()));
				} else {
					QTextStream in(&file);
					QApplication::setOverrideCursor(Qt::WaitCursor);
					textEdit->setPlainText(in.readAll());
					QApplication::restoreOverrideCursor();
					statusBar()->showMessage(QString("Loaded file: %1").arg(filename), 2000);
					loadingSucceded = true;
				}
			}


			QString displayName = filename;
			if (displayName.isEmpty()) {
				// Find a new name
				displayName = "Unnamed";
				QString suggestedName = displayName;

				bool unique = false;
				int counter = 1;
				while (!unique) {
					unique = true;
					for (int i = 0; i < tabInfo.size(); i++) {
						if (tabInfo[i].filename == suggestedName) {
							INFO("equal");
							unique = false;
							break;
						}	
					}
					if (!unique) suggestedName = displayName + " " + QString::number(counter++);
				}
				displayName = suggestedName;
			}

			stackedTextEdits->addWidget(textEdit);

			if (loadingSucceded) {
				tabInfo.append(TabInfo(displayName, textEdit, false, true));
				setRecentFile(filename);

			} else {
				tabInfo.append(TabInfo(displayName, textEdit, true));
			}

			QString tabTitle = QString("%1%3").arg(strippedName(displayName)).arg(!loadingSucceded? "*" : "");
			tabBar->setCurrentIndex(tabBar->addTab(strippedName(tabTitle)));

			connect(textEdit->document(), SIGNAL(contentsChanged()), this, SLOT(documentWasModified()));
		}

		void MainWindow::tabChanged(int index) {
			if (index > tabInfo.size()) return;
			if (index < 0) return;

			TabInfo t = tabInfo[index];
			QString tabTitle = QString("%1%3").arg(strippedName(t.filename)).arg(t.unsaved ? "*" : "");
			setWindowTitle(QString("%1 - %2").arg(tabTitle).arg("Structure Synth"));
			stackedTextEdits->setCurrentWidget(t.textEdit);
			tabBar->setTabText(tabBar->currentIndex(), tabTitle);
		}

		void MainWindow::closeTab() {
			int index = tabBar->currentIndex();
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 
			closeTab(index);
		}

		void MainWindow::closeTab(int index) {		
			TabInfo t = tabInfo[index];
			if (t.unsaved) {
				int answer = QMessageBox::warning(this, QString("Unsaved changes"), "Close this tab without saving changes?", "OK", "Cancel");
				if (answer == 1) return;
			}

			tabInfo.remove(index);
			tabBar->removeTab(index);

			stackedTextEdits->removeWidget(t.textEdit);
			delete(t.textEdit); // ?
		}

		void MainWindow::launchSfHome() {
			INFO("Launching web browser...");
			bool s = QDesktopServices::openUrl(QUrl("http://structuresynth.sourceforge.net/"));
			if (!s) WARNING("Failed to open browser...");
		}

		void MainWindow::launchReferenceHome() {
			INFO("Launching web browser...");
			bool s = QDesktopServices::openUrl(QUrl("http://structuresynth.sourceforge.net/reference.php"));
			if (!s) WARNING("Failed to open browser...");
		}

		void MainWindow::launchGallery() {
			INFO("Launching web browser...");
			bool s = QDesktopServices::openUrl(QUrl("http://flickr.com/groups/structuresynth/"));
			if (!s) WARNING("Failed to open browser...");
		}

		void MainWindow::saveImage(QImage image) {
			QList<QByteArray> a = QImageWriter::supportedImageFormats();
			QStringList allowedTypesFilter;
			QStringList allowedTypes;
			for (int i = 0; i < a.count(); i++) {
				allowedTypesFilter.append("*."+a[i]);
				allowedTypes.append(a[i]);
			}
			QString filter = "Image Files (" + allowedTypesFilter.join(" ")+")";

			QString filename = QFileDialog::getSaveFileName(this, "Save Screenshot As...", QString(), filter);
			if (filename.isEmpty()) {
				INFO("User cancelled save...");
				return;
			}

			QString ext = filename.section(".", -1).toLower();
			if (!allowedTypes.contains(ext)) {
				WARNING("Invalid image extension.");
				WARNING("File must be one of the following types: " + allowedTypes.join(","));
				return;
			}

			bool succes = image.save(filename);
			if (succes) {
				INFO("Saved screenshot as: " + filename);
			} else {
				WARNING("Save failed! Filename: " + filename);
			}
		}

		void MainWindow::makeScreenshot() {
			QImage image = engine->getScreenShot();
			saveImage(image);
		}

		void MainWindow::setSeed(int randomSeed) {
			seedSpinBox->setValue(randomSeed);
		}

		int MainWindow::getSeed() {
			return seedSpinBox->value();
		};

		void MainWindow::insertCameraSettings() {
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 


			getTextEdit()->insertPlainText(getCameraSettings());
			INFO("Camera settings are now pasted into the script window.");
			INFO("Remember to clear previous 'set' commands if necessary.");
		}



		QString MainWindow::getScriptWithSettings(QString filename) {
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return ""; } 

			QString s = "// Autogenerated script for file: " + filename + "\n";
			s += "// Generated " + QDateTime::currentDateTime().toString("MMMM dd yyyy hh:mm:ss") + "\n";
			s += QString("// Original window size: %1x%2").arg(engine->width()).arg(engine->height()) + "\n\n";
			s += getCameraSettings() + "\n";
			s += QString("Set seed %1\n\n").arg(getSeed());
			s += QString("// Original script:\n\n");
			s += getTextEdit()->toPlainText();
			return s;
		}

		QString MainWindow::getCameraSettings() {

			Vector3f translation = engine->getTranslation();
			Matrix4f rotation = engine->getRotation();
			Vector3f pivot = engine->getPivot();
			double scale = engine->getScale();

			QStringList sl;
			sl << "// Camera settings. Place these before first rule call." 
				<< QString("set translation %1").arg(translation.toString())
				<< QString("set rotation %1").arg(rotation.toStringAs3x3())
				<< QString("set pivot %1").arg(pivot.toString())
				<< QString("set scale %1").arg(scale);
			return sl.join("\n");

		}


		void MainWindow::templateRenderToFile()
		{
			QString fileName = QFileDialog::getSaveFileName(this, tr("Save As"), "output.txt");
			if (fileName.isEmpty()) {
				INFO("User cancelled.");
				return;
			}

			templateRender(fileName);
		}

		void MainWindow::templateRender()
		{
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 

			templateRender(""); // Renders to clip board when file name is empty.
		}

		void MainWindow::templateRender(const QString& fileName)
		{
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 

			QAction *action = qobject_cast<QAction *>(sender());
			if (action) {
				QDir d(getTemplateDir());
				QString templateFileName = d.absoluteFilePath(action->data().toString());
				INFO("Starting Template Renderer: " + fileName);
				try {
					Template myTemplate(templateFileName);
					templateRender(fileName, &myTemplate, "");
				} catch (Exception& er) {
					WARNING(er.getMessage());
				}
			} else {
				WARNING("No data!");
			}
		}

		void MainWindow::templateRender(const QString& fileName, Model::Rendering::Template* myTemplate, QString inputText,
			int width, int height, bool postModify)
		{
			RandomStreams::SetSeed(getSeed());
			if (inputText.isEmpty()) inputText = getTextEdit()->toPlainText();
			INFO(QString("Random seed: %1").arg(getSeed()));
			try {
				if (width == 0) width = engine->width();
				if (height == 0) height = engine->height();
				TemplateRenderer rendering(*myTemplate);
				Vector3f cameraRight=  (engine->getCameraPosition()-engine->getCameraTarget()).cross(engine->getCameraUp());
				cameraRight.normalize();
				rendering.setCamera(
					engine->getCameraPosition(), 
					engine->getCameraUp().normalized(), 
					cameraRight,
					engine->getCameraTarget(),
					width, height, width/(double)height, engine->getFOV());

				rendering.setBackgroundColor(engine->getBackgroundColor());
				rendering.begin(); 

				Preprocessor pp;
				QString out = pp.Process(inputText, getSeed());
				bool showGUI = false;
				out = variableEditor->updateFromPreprocessor(&pp, out, &showGUI);
				editorDockWidget->setHidden(!showGUI);

				Tokenizer tokenizer(out);
				EisenParser e(&tokenizer);
				INFO("Started Eisenstein engine...");
				RuleSet* rs = e.parseRuleset();

				rs->resolveNames();

				rs->dumpInfo();

				Builder b(&rendering, rs, true);
				b.build();
				rendering.end();

				if (b.seedChanged()) {
					setSeed(b.getNewSeed());
					INFO(QString("Builder changed seed to: %1").arg(b.getNewSeed()));
				} 

				QString output = rendering.getOutput();

				if (postModify) {
					// Post modify output.
					QDialog* d = new QDialog(this);
					QVBoxLayout* vl = new QVBoxLayout(d);

					d->resize(800,600);
					QTextEdit* te = new QTextEdit(d);
					vl->addWidget(te);
					te->setText(output);

					QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok
						| QDialogButtonBox::Cancel);
					vl->addWidget(buttonBox);						

					connect(buttonBox, SIGNAL(accepted()), d, SLOT(accept()));
					connect(buttonBox, SIGNAL(rejected()), d, SLOT(reject()));

					if (d->exec() == QDialog::Accepted) {
						output = te->toPlainText();
					} else {
						WARNING("User cancelled...");
						return;
					}
				}

				if (fileName.isEmpty()){
					QClipboard *clipboard = QApplication::clipboard();
					clipboard->setText(output); 
					INFO("Done...");
					INFO("Script is now copied to the clipboard");
				} else {
					QFile file(fileName);
					INFO("Writing to file: " + QFileInfo(file).absoluteFilePath());
					if (!file.open(QFile::WriteOnly | QFile::Text)) {
						QMessageBox::warning(this, tr("Structure Synth"),
							tr("Cannot write file %1:\n%2.")
							.arg(fileName)
							.arg(file.errorString()));
						return;
					}

					QTextStream out(&file);
					QApplication::setOverrideCursor(Qt::WaitCursor);
					out << output;
					QApplication::restoreOverrideCursor();
					INFO("File saved.");
				}

				delete(rs);
				rs = 0;


			} catch (Exception& er) {
				WARNING(er.getMessage());
			}

		}


		void MainWindow::parseJavaScript(QString scripture, QString dir) {
			JavaScriptParser jsp(this, statusBar());
			jsp.parse(scripture, dir);
		}

		void MainWindow::copy() {
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 
			getTextEdit()->copy();
		}


		void MainWindow::cut() {
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 
			getTextEdit()->cut();
		}

		void MainWindow::paste() {
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 
			getTextEdit()->paste();
		}

		void MainWindow::dragEnterEvent(QDragEnterEvent *ev)
		{
			if (ev->mimeData()->hasUrls()) {
				ev->acceptProposedAction();
			} else {
				INFO("Cannot accept MIME object: " + ev->mimeData()->formats().join(" - "));
			}
		}

		void MainWindow::dropEvent(QDropEvent *ev) {
			if (ev->mimeData()->hasUrls()) {
				QList<QUrl> urls = ev->mimeData()->urls();
				for (int i = 0; i < urls.size() ; i++) {
					INFO("Loading: " + urls[i].toLocalFile());
					loadFile(urls[i].toLocalFile());
				}
			} else {
				INFO("Cannot accept MIME object: " + ev->mimeData()->formats().join(" - "));
			}
		}

		void MainWindow::templateExport() {
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 

			// We must parse first...
			RuleSet* rs = 0;
			QStringList primitives;
			try {	
				Preprocessor pp;
				QString out = pp.Process(getTextEdit()->toPlainText(), getSeed());
				bool showGUI = false;
				out = variableEditor->updateFromPreprocessor(&pp, out, &showGUI);
				editorDockWidget->setHidden(!showGUI);

				Tokenizer tokenizer(out);
				EisenParser e(&tokenizer);
				INFO("Checking EisenScript...");
				rs = e.parseRuleset();
				primitives = rs->resolveNames();

				rs->dumpInfo();
			} catch (Exception& er) {
				WARNING("Could not export, because script could not be parsed:");
				WARNING(er.getMessage());
				return; // something went wrong...
			}

			TemplateExportDialog dialog(this, primitives);
			dialog.setDefaultSize(engine->width(), engine->height());
			dialog.setTemplatePath(getTemplateDir());
			dialog.exec();
		}

		void MainWindow::setRecentFile(const QString &fileName)
		{
			QSettings settings;

			QStringList files = settings.value("recentFileList").toStringList();
			files.removeAll(fileName);
			files.prepend(fileName);
			while (files.size() > MaxRecentFiles) files.removeLast();

			settings.setValue("recentFileList", files);

			int numRecentFiles = qMin(files.size(), (int)MaxRecentFiles);

			for (int i = 0; i < numRecentFiles; ++i) {
				QString text = tr("&%1 %2").arg(i + 1).arg(QFileInfo(files[i]).fileName());
				recentFileActions[i]->setText(text);
				QString absPath = QFileInfo(files[i]).absoluteFilePath();
				recentFileActions[i]->setData(absPath);
				recentFileActions[i]->setVisible(true);
			}

			for (int j = numRecentFiles; j < MaxRecentFiles; ++j) recentFileActions[j]->setVisible(false);

			recentFileSeparator->setVisible(numRecentFiles > 0);
		}

		namespace {
			// Very simple complex numbers.
			class Complex {
			public:
				Complex(double r, double i) : r(r), i(i) {};
				Complex() : r(0), i(0) {};

				Complex operator*(const double& rhs) const { return Complex(r*rhs, i*rhs); }
				Complex operator*(const Complex& rhs) const { return Complex(r*rhs.r-i*rhs.i,r*rhs.i+i*rhs.r); }
				Complex operator+(const Complex& rhs) const { return Complex(r+rhs.r,i+rhs.i); }
				double sqrLength() const { return r*r+i*i; } 
				double length2() const { return fabs(r)>fabs(i) ? fabs(r) : fabs(i); } 
				Complex raisedTo(Complex power) const {
					double a = r; double b = i;
					double c = power.r; double d = power.i;
					double p = sqrt(a*a + b*b);
					double t = atan2(b,a); // is this correct quadrant?
					double factor = pow(p,c)*exp(-d*t);
					double r = factor*cos(c*t + d*log(p));
					double i = factor*sin(c*t + d*log(p));
					return Complex(r,i);
				}
				QString toString() { return QString("(%1,%2)").arg(r).arg(i); } 

				double r;
				double i;
			};
		}

		namespace {
			// Could we make a match("Syntax: %1, %2, %3").arg(double*).arg(int*).arg();
			int match(QString s, QString m, double* a, double* b = 0, double* c = 0, double* d = 0) {
				m.replace("(", "\\(");
				m.replace(")", "\\)");
				m.replace("^", "\\^");
				m.replace("*", "\\*");
				m.replace("@", "([-+]?[0-9]*\\.?[0-9]+)"); // 
				QRegExp rx("^" + m + "$");
				int i = rx.indexIn(s);
				if (i >= 0) {
					QStringList list = rx.capturedTexts();
					if (list.count()>1 && a) (*a) = list[1].toDouble();
					if (list.count()>2 && b) (*b) = list[2].toDouble();
					if (list.count()>3 && c) (*c) = list[3].toDouble();
					if (list.count()>4 && d) (*d) = list[4].toDouble();
					//INFO(QString("Captured %1 on %2, %3").arg(list.count()).arg(m).arg(s));
					return list.count()-1;
				} 
				return -1;
			}

			struct Term {
				Term() {};
				Term(Complex f, Complex e) : factor(f), exponent(e) {};
				Complex factor;
				Complex exponent;
			};
		}

		void MainWindow::raytrace() {
			INFO("Hint: use the 'set raytracer::size [0x800]' command to set the dimensions. use 'set raytracer::samples 10' to control quality."); 
			disableAllExcept(progressBox);
			RayTracer rt(engine, progressBox, false);
			QImage im = rt.calculateImage(0,0);
			enableAll();
			PreviewWindow pd(this, im);
			pd.exec();
		}

		PreviewWindow::PreviewWindow(MainWindow* mainWindow, QImage im) : QDialog(mainWindow), mainWindow(mainWindow), image(im) {
			QVBoxLayout *layout = new QVBoxLayout(this);
			setLayout(layout);

			QHBoxLayout* hlayout = new QHBoxLayout();

			QPushButton* pb = new QPushButton(this);
			pb->setText("Save as Bitmap...");
			connect(pb, SIGNAL(clicked()), this, SLOT(saveImage()));
			hlayout->addWidget(pb);

			QPushButton* pb2 = new QPushButton(this);
			pb2->setText("Close");
			connect(pb2, SIGNAL(clicked()), this, SLOT(close()));
			hlayout->addWidget(pb2);

			hlayout->addStretch(10);
			layout->addLayout(hlayout);

			QLabel* lb = new QLabel(this);
			lb->setPixmap(QPixmap::fromImage(image));
			lb->resize(im.width(), im.height());
			layout->addWidget(lb);
			resize(lb->size());
		};

		void PreviewWindow::saveImage() {
			mainWindow->saveImage(image);
		}

		namespace {
			class ObjDialog : public QDialog {
			public:
				ObjDialog(QWidget* parent) : QDialog(parent) {
					QSettings s;

					setWindowTitle("Obj Export Settings");
					QVBoxLayout* vl = new QVBoxLayout(this);
					
					groupByTaggingCheckBox = new QCheckBox("Group by Tagging", this);
					vl->addWidget(groupByTaggingCheckBox);
					bool groupByTagging = s.value("ObjDialog/groupByTagging", true).toBool();
					groupByTaggingCheckBox->setChecked(groupByTagging);
					
					groupByColorCheckBox = new QCheckBox("Group by Color", this);
					bool groupByColor = s.value("ObjDialog/groupByColor", false).toBool();
					groupByColorCheckBox->setChecked(groupByColor);
					vl->addWidget(groupByColorCheckBox);
					
					QHBoxLayout* hl = new QHBoxLayout(this);
					
					QLabel* l = new QLabel(QString("Sphere resolution: Latitude: "), this);
					spinBox1 = new QSpinBox(this);
					spinBox1->setMinimum(2);
					spinBox1->setMaximum(40);
					int latResolution = s.value("ObjDialog/latResolution", 10).toInt();
					spinBox1->setValue(latResolution);
					QLabel* l2 = new QLabel(QString(" divs, Longitude: "), this);
					spinBox2 = new QSpinBox(this);
					spinBox2->setMinimum(2);
					spinBox2->setMaximum(40);
					int longResolution = s.value("ObjDialog/longResolution", 10).toInt();					
					spinBox2->setValue(longResolution);
					QLabel* l3 = new QLabel(QString(" divs."));
					hl->addWidget(l);
					hl->addWidget(spinBox1);
					hl->addWidget(l2);
					hl->addWidget(spinBox2);
					hl->addWidget(l3);
					vl->addLayout(hl);

					QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
					vl->addWidget(buttonBox);						

					connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
					connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

					vl->addWidget(buttonBox);
				};

				~ObjDialog() {
					QSettings s;
					s.setValue("ObjDialog/groupByTagging", groupByTaggingCheckBox->isChecked());
					s.setValue("ObjDialog/groupByColor", groupByColorCheckBox->isChecked());
					s.setValue("ObjDialog/latResolution", spinBox1->value());
					s.setValue("ObjDialog/longResolution", spinBox2->value());
				}
				
				QSpinBox* spinBox1;
				QSpinBox* spinBox2;
				QCheckBox* groupByTaggingCheckBox;
				QCheckBox* groupByColorCheckBox ;

			};
		}

		void MainWindow::insertText() {
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 

			QString text = ((QAction*)sender())->text();
			getTextEdit()->insertPlainText(text.section("//",0,0)); // strip comments
		}
		void MainWindow::exportToObj()
		{
			if (tabBar->currentIndex() == -1) { WARNING("No open tab"); return; } 

			ObjDialog od(this);
			if (od.exec() == QDialog::Rejected) return;
			int dt = od.spinBox1->value();
			int dp = od.spinBox2->value();

			QString fileName = QFileDialog::getSaveFileName(this, tr("Save As"), "output.obj");
			if (fileName.isEmpty()) {
				INFO("User cancelled.");
				return;
			}

			RandomStreams::SetSeed(getSeed());
			QString inputText = getTextEdit()->toPlainText();

			INFO(QString("Random seed: %1").arg(getSeed()));
			try {
				ObjRenderer renderer(dt,dp, od.groupByTaggingCheckBox->isChecked(), od.groupByColorCheckBox->isChecked() );
				renderer.begin(); 

				Preprocessor pp;
				QString out = pp.Process(inputText, getSeed());
				bool showGUI = false;
				out = variableEditor->updateFromPreprocessor(&pp, out, &showGUI);
				editorDockWidget->setHidden(!showGUI);

				Tokenizer tokenizer(out);
				EisenParser e(&tokenizer);
				INFO("Started Eisenstein engine...");
				RuleSet* rs = e.parseRuleset();
				rs->resolveNames();
				rs->dumpInfo();

				Builder b(&renderer, rs, true);
				b.build();
				renderer.end();


				QFile file(fileName);
				INFO("Writing to file: " + QFileInfo(file).absoluteFilePath());
				if (!file.open(QFile::WriteOnly | QFile::Text)) {
					QMessageBox::warning(this, tr("Structure Synth"),
						tr("Cannot write file %1:\n%2.")
						.arg(fileName)
						.arg(file.errorString()));
					return;
				}

				QTextStream out2(&file);
				QApplication::setOverrideCursor(Qt::WaitCursor);
				renderer.writeToStream(out2);
				QApplication::restoreOverrideCursor();
				INFO("File saved.");
				delete(rs);
			} catch (Exception& er) {
				WARNING(er.getMessage());
			}

		}


	}

}

