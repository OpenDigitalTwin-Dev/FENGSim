#pragma once

#include <QString>
#include <QStatusBar>
#include "../GUI/MainWindow.h"


namespace StructureSynth {
	namespace JavaScriptSupport {	

		/// Responsible for setting up the JavaScript environment and parsing
		class JavaScriptParser {
		public:
			JavaScriptParser(StructureSynth::GUI::MainWindow* mainWindow, QStatusBar* statusBar);
			~JavaScriptParser();
			
			void parse(QString input, QString dir);
		private:
			StructureSynth::GUI::MainWindow* mainWindow;
			QStatusBar* statusBar;
		};

	}
}

