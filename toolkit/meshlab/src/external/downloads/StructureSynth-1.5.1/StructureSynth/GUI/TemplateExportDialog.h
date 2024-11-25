#pragma once

#include <QString>
#include <QVector>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QFrame>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QTabWidget>
#include <QtGui/QTableWidget>
#include <QtGui/QTextBrowser>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

#include "../Model/Rendering/TemplateRenderer.h"

namespace StructureSynth {
	namespace GUI {


		class MainWindow; // Forward declaration.

		/// The Variable Editor window.
		class TemplateExportDialog : public QDialog {
			Q_OBJECT
		public:
			TemplateExportDialog(MainWindow* parent, QStringList primitives);
			~TemplateExportDialog();


			void setTemplatePath(QString templatePath);
			void setDefaultSize(int width, int height);
			void changeFileNameExtension(QString extension);
		

		public slots:
			void multiplySize(double d);
			void halfSize();
			void doubleSize();	
			void defaultSize();	
			void accept();
			void reject();
			void templateChanged(const QString &);
			void saveModifications();
			void undo();
			void textChanged();
			void tabChanged(int i);
		

			void lockAspectChanged();
			void heightChanged(int);
			void widthChanged(int);
			void selectFileName();
			void uniqueToggled(bool);
			void updateUniqueFileName(const QString &);
			

			void fileRadioButtonToggled(bool);
			void changeTemplatePath();

		protected:
			void retranslateUi();

			void setupUi();

		private:
			void setAspectLabel(double ratio) ;
			Model::Rendering::Template currentTemplate;

			QStringList primitives;
			QString path;
			QVBoxLayout *verticalLayout;
			QHBoxLayout *horizontalLayout;
			QLabel *label;
			QComboBox *templateComboBox;
			QPushButton *templatePathButton;
			QTabWidget *tabWidget;
			QWidget *settingstab;
			QVBoxLayout *verticalLayout_3;
			QLabel *descriptionLabel;
			QTextBrowser *descriptionTextBrowser;
			QLabel *primitivesLabel;
			QTableWidget *primitivesTableWidget;
			QGroupBox *templateOutputGroupBox;
			QVBoxLayout *verticalLayout_2;
			QHBoxLayout *horizontalLayout_2;
			QRadioButton *fileRadioButton;
			QLineEdit *fileNameLineEdit;
			QPushButton *filePushButton;
			QHBoxLayout *horizontalLayout_3;
			QSpacerItem *horizontalSpacer;
			QCheckBox *uniqueCheckBox;
			QCheckBox *autoSaveCheckBox;
			QRadioButton *clipboardRadioButton;
			QGroupBox *postProcessingGroupBox;
			QVBoxLayout *verticalLayout_4;
			QCheckBox *runAfterCheckBox;
			QHBoxLayout *horizontalLayout_4;
			QSpacerItem *horizontalSpacer_2;
			QLineEdit *afterCommandLineEdit;
			QWidget *advancedTab;
			QVBoxLayout *verticalLayout_5;
			QHBoxLayout *horizontalLayout_5;
			QLabel *label_4;
			QSpinBox *widthSpinBox;
			double aspectRatio;
			QLabel *label_5;
			QSpinBox *heightSpinBox;
			QFrame *line;
			QCheckBox *lockAspectRatioCheckBox;
			QSpacerItem *horizontalSpacer_4;
			QLabel *modifyTemplateLabel;
			QTextEdit *templateTextEdit;
			QHBoxLayout *horizontalLayout_6;
			QSpacerItem *horizontalSpacer_3;
			QPushButton *saveModificationsButton;
			QPushButton *undoButton;
			QCheckBox *	modifyOutputCheckBox;
			QDialogButtonBox *buttonBox;
			QString modifiedTemplate;
			MainWindow* mainWindow;
			QString uniqueFileName;
			int screenWidth;
			int screenHeight;
		};


	}
}

