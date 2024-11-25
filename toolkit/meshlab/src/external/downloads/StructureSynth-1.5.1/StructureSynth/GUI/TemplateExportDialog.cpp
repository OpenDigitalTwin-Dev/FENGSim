#include "TemplateExportDialog.h"
#include "MainWindow.h"

#include <QPushButton>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QDir>
#include <QSplitter>
#include <QFileDialog>
#include <QProcess>
#include <QSyntaxHighlighter>

#include "../../SyntopiaCore/Logging/ListWidgetLogger.h"
#include "../../SyntopiaCore/Misc/Persistence.h"
#include "../Model/Rendering/TemplateRenderer.h"
#include "../../SyntopiaCore/Exceptions/Exception.h"

using namespace SyntopiaCore::Logging;
using namespace SyntopiaCore::Exceptions;
using namespace SyntopiaCore::Misc;

namespace StructureSynth {
	using namespace Model::Rendering;

	namespace GUI {


		namespace {

			class XmlHighlighter : public QSyntaxHighlighter {
			public:

				XmlHighlighter(QTextEdit* e) : QSyntaxHighlighter(e) {
					elementFormat.setForeground(Qt::blue);
					attributeFormat.setForeground(Qt::red);
					commentFormat.setForeground(Qt::darkGreen);
					cdataFormat.setForeground(Qt::darkBlue);
				};

				void highlightBlock(const QString &text)
				{

					
					if (previousBlockState() != 1 && currentBlockState() == 1) {
						// This line was previously a multi-line start 
						if (!text.contains("<!--")) setCurrentBlockState(0);
					}

					if (previousBlockState() == 1) {
						// Part of multi-line comment. Skip the rest...
						if (!text.contains("-->")) {
							setFormat(0, text.length(), commentFormat);
							setCurrentBlockState(1);
							return;
						}
					}

					if (previousBlockState() != 2 && currentBlockState() == 2) {
						if (!text.contains("<![CDATA[")) setCurrentBlockState(0);
					}

					if (previousBlockState() == 2) {
						// Part of multi-line comment. Skip the rest...
						if (!text.contains("]]>")) {
							setFormat(0, text.length(), cdataFormat);
							setCurrentBlockState(2);
							return;
						}
					}

					// Line parsing
					QString current;
					bool inElement = false;
					bool expectingAttribute = false;
					bool insideBracket = false;
					bool inQuote = false;
					for (int i = 0; i < text.length(); i++) {
							
						if ((i >= 3) && (text.mid(i-3,4)=="<!--")) {
							// Multi-line comment begins
							setFormat(i-3, text.length()-(i-3), commentFormat);
							setCurrentBlockState(1);
							insideBracket = true;
						}

						if ((i >= 2) && (text.mid(i-2,3)=="-->")) {
							// Multi-line comment ends
							setFormat(i-2, text.length()-(i-2), standardFormat);
							if (currentBlockState() != 0) {
								setCurrentBlockState(0);
							}
							insideBracket = false;
						
							continue;
						}

						if ((i >= 8) && (text.mid(i-8,9)=="<![CDATA[")) {
							// Multi-line comment begins
							setFormat(i-8, text.length()-(i-8), cdataFormat);
							setCurrentBlockState(2);
							insideBracket = true;
						}

						if ((i >= 2) && (text.mid(i-2,3)=="]]>")) {
							// Multi-line comment ends
							setFormat(i-2, text.length()-(i-2), standardFormat);
							if (currentBlockState() != 0) {
								setCurrentBlockState(0);
							}
							insideBracket = false;
						
							continue;
						}
						
						if (insideBracket) continue;
				
						current += text.at(i);
						

						if (text.at(i) == '"') {
							inQuote = !inQuote;
						}
						
						if (!inQuote) {
							if (text.at(i) == '<') {
								inElement = true;
								expectingAttribute = false; 
							}

							if (text.at(i) == ' ') expectingAttribute = true;
							if (text.at(i) == '=') expectingAttribute = false;
						}
						
						
						if (inElement) {
							if (expectingAttribute) {
								setFormat(i, i+1, attributeFormat);
							} else {
								setFormat(i, i+1, elementFormat);
							}
						} else {
							setFormat(i, i+1, standardFormat);
						}

						if (text.at(i) == '>') inElement = false;

						
					}

				}; 
			private:
				QTextCharFormat elementFormat;
				QTextCharFormat cdataFormat;
				QTextCharFormat standardFormat;
				QTextCharFormat attributeFormat;
				QTextCharFormat commentFormat;

			};
		};

		TemplateExportDialog::TemplateExportDialog(MainWindow* parent, QStringList primitives) :  QDialog(parent), primitives(primitives) {
			// 'primitives' contain a list of used primitives.
			// We add 'begin' and 'end' since these are always used.
			this->primitives.append("begin");
			this->primitives.append("end");
			setupUi();
			retranslateUi();
			mainWindow = parent;
			uniqueFileName = "";
			
		}

		TemplateExportDialog::~TemplateExportDialog() {
			// Persist (should only be done on OK?)
			
		}

		void TemplateExportDialog::setDefaultSize(int width, int height) {
			
			screenWidth = width;
			screenHeight = height;

			// We will check if the window size has been changed since last 
			// invocation if this dialog.
			// If it has changed, we will overwrite the settings (to keep Aspect Ratio).
			bool sizeChanged = false;
			if (Persistence::Contains("TemplateExportDialog.screenWidth")) {
				int sw = Persistence::Get("TemplateExportDialog.screenWidth").toInt();
				int sh = Persistence::Get("TemplateExportDialog.screenHeight").toInt();

				if (sw != screenWidth || sh != screenHeight) {
					sizeChanged = true;
				}
			}
			Persistence::Put("TemplateExportDialog.screenWidth", width);
			Persistence::Put("TemplateExportDialog.screenHeight", height);


			heightSpinBox->blockSignals(true);
			widthSpinBox->blockSignals(true);
			lockAspectRatioCheckBox->blockSignals(true);
			if (sizeChanged) {
				INFO("OpenGL window size has been changed. Setting output width/height to window size.");
				heightSpinBox->setValue(height);
				widthSpinBox->setValue(width);
				lockAspectRatioCheckBox->setChecked(true);
			} else {
				heightSpinBox->setValue(height);
				Persistence::Restore(heightSpinBox);
				widthSpinBox->setValue(width);
				Persistence::Restore(widthSpinBox);
				lockAspectRatioCheckBox->setChecked(true);
				Persistence::Restore(lockAspectRatioCheckBox);
			}
			heightSpinBox->blockSignals(false);
			widthSpinBox->blockSignals(false);
			lockAspectRatioCheckBox->blockSignals(false);
			
			
			aspectRatio = widthSpinBox->value()/(double)heightSpinBox->value();
			setAspectLabel(aspectRatio);
		}

		void TemplateExportDialog::setAspectLabel(double ratio) {
			lockAspectRatioCheckBox->setText(QString("Lock ratio (AR = %1)").arg(ratio, 0, 'f', 3));
		}

		void TemplateExportDialog::lockAspectChanged() {
		}

		void TemplateExportDialog::multiplySize(double d) {	
			if (lockAspectRatioCheckBox->isChecked()) {
				heightSpinBox->setValue((int)(d*heightSpinBox->value()));
			} else {
				heightSpinBox->setValue((int)(d*heightSpinBox->value()));
				widthSpinBox->setValue((int)(d*widthSpinBox->value()));		
			}
			heightChanged(0);
		}
		
		

		void TemplateExportDialog::halfSize() {	
			multiplySize(0.5);
		}

		void TemplateExportDialog::doubleSize() {	
			multiplySize(2.0);
		}

		void TemplateExportDialog::defaultSize() {	
			heightSpinBox->blockSignals(true);
			widthSpinBox->blockSignals(true);
			lockAspectRatioCheckBox->blockSignals(true);
			heightSpinBox->setValue(screenHeight);
			widthSpinBox->setValue(screenWidth);
			lockAspectRatioCheckBox->setChecked(true);
			heightSpinBox->blockSignals(false);
			widthSpinBox->blockSignals(false);
			lockAspectRatioCheckBox->blockSignals(false);
			aspectRatio = widthSpinBox->value() / (double)heightSpinBox->value();
			
			setAspectLabel(aspectRatio);
		
		}

		void TemplateExportDialog::heightChanged(int) {			
			if (lockAspectRatioCheckBox->isChecked()) {
				widthSpinBox->blockSignals(true);
				widthSpinBox->setValue((int)(aspectRatio*heightSpinBox->value()));
				widthSpinBox->blockSignals(false);
			} else {
				aspectRatio = widthSpinBox->value() / (double)heightSpinBox->value();
				setAspectLabel(aspectRatio);
			}
		}

		void TemplateExportDialog::widthChanged(int) {
			if (lockAspectRatioCheckBox->isChecked()) {
				heightSpinBox->blockSignals(true);
				heightSpinBox->setValue((int)(widthSpinBox->value()/aspectRatio));
				heightSpinBox->blockSignals(false);			
			} else {
				aspectRatio = widthSpinBox->value() / (double)heightSpinBox->value();
				setAspectLabel(aspectRatio);
			}
		}

		void TemplateExportDialog::setupUi()
		{
			autoSaveCheckBox = 0;
			if (objectName().isEmpty())
				setObjectName(QString::fromUtf8("TemplateExportDialog"));
			resize(544, 600);
			setSizeGripEnabled(true);
			setModal(true);
			verticalLayout = new QVBoxLayout(this);
			verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
			horizontalLayout = new QHBoxLayout();
			horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
			label = new QLabel(this);
			label->setObjectName(QString::fromUtf8("label"));

			horizontalLayout->addWidget(label);

			templateComboBox = new QComboBox(this);
			templateComboBox->setObjectName(QString::fromUtf8("TemplateExportDialog.templateComboBox"));
			QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
			sizePolicy.setHorizontalStretch(0);
			sizePolicy.setVerticalStretch(0);
			sizePolicy.setHeightForWidth(templateComboBox->sizePolicy().hasHeightForWidth());
			templateComboBox->setSizePolicy(sizePolicy);
			connect(templateComboBox, SIGNAL(currentIndexChanged(const QString &)), 
				this, SLOT(templateChanged(const QString &)));

			if (!Persistence::Contains(templateComboBox->objectName())) {
				Persistence::Put(templateComboBox->objectName(), "Sunflow-Colored");
			}
			
			

			horizontalLayout->addWidget(templateComboBox);

			templatePathButton = new QPushButton(this);
			templatePathButton->setObjectName(QString::fromUtf8("templatePathButton"));

			horizontalLayout->addWidget(templatePathButton);
			connect(templatePathButton, SIGNAL(clicked()), this, SLOT(changeTemplatePath()));


			verticalLayout->addLayout(horizontalLayout);

			tabWidget = new QTabWidget(this);
			tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
			settingstab = new QWidget();
			settingstab->setObjectName(QString::fromUtf8("settingstab"));
			settingstab->setAutoFillBackground(false);

			verticalLayout_3 = new QVBoxLayout(settingstab);
			verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));

			QSplitter* splitter = new QSplitter(settingstab);
			verticalLayout_3->addWidget(splitter);


			QWidget* box1 = new QWidget(splitter);
			QVBoxLayout* box1Layout = new QVBoxLayout(box1);

			QWidget* box2 = new QWidget(splitter);
			QVBoxLayout* box2Layout = new QVBoxLayout(box2);

			splitter->addWidget(box1);
			splitter->addWidget(box2);
			QList<int> list;
			list.append(200);
			list.append(100);
			splitter->setSizes(list);



			descriptionLabel = new QLabel(settingstab);
			descriptionLabel->setObjectName(QString::fromUtf8("descriptionLabel"));
			box1Layout->addWidget(descriptionLabel);
			descriptionTextBrowser = new QTextBrowser(settingstab);
			descriptionTextBrowser->setObjectName(QString::fromUtf8("descriptionTextBrowser"));
			box1Layout->addWidget(descriptionTextBrowser);


			primitivesLabel = new QLabel(settingstab);
			primitivesLabel->setObjectName(QString::fromUtf8("primitivesLabel"));

			box2Layout->addWidget(primitivesLabel);

			primitivesTableWidget = new QTableWidget(settingstab);
			primitivesTableWidget->horizontalHeader()->setStretchLastSection(true);
			primitivesTableWidget->setObjectName(QString::fromUtf8("primitivesTableWidget"));

			box2Layout->addWidget(primitivesTableWidget);

			templateOutputGroupBox = new QGroupBox(settingstab);
			templateOutputGroupBox->setObjectName(QString::fromUtf8("templateOutputGroupBox"));
			templateOutputGroupBox->setFlat(true);
			verticalLayout_2 = new QVBoxLayout(templateOutputGroupBox);
			verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
			horizontalLayout_2 = new QHBoxLayout();
			horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
			fileRadioButton = new QRadioButton(templateOutputGroupBox);
			fileRadioButton->setObjectName(QString::fromUtf8("fileRadioButton"));
			fileRadioButton->setChecked(true);
			connect(fileRadioButton, SIGNAL(toggled(bool)), this, SLOT(fileRadioButtonToggled(bool)));
			horizontalLayout_2->addWidget(fileRadioButton);

			fileNameLineEdit = new QLineEdit(templateOutputGroupBox);
			fileNameLineEdit->setObjectName(QString::fromUtf8("TemplateExportDialog.fileNameLineEdit"));
			QString initialFileName = QFileInfo("output.txt").absoluteFilePath();
			fileNameLineEdit->setText(initialFileName);
			Persistence::Restore(fileNameLineEdit);
			connect(fileNameLineEdit, SIGNAL(textChanged(const QString &)),this, SLOT(updateUniqueFileName(const QString &)));


			horizontalLayout_2->addWidget(fileNameLineEdit);

			filePushButton = new QPushButton(templateOutputGroupBox);
			filePushButton->setObjectName(QString::fromUtf8("filePushButton"));
			QSizePolicy sizePolicy1(QSizePolicy::Maximum, QSizePolicy::Fixed);
			sizePolicy1.setHorizontalStretch(0);
			sizePolicy1.setVerticalStretch(0);
			sizePolicy1.setHeightForWidth(filePushButton->sizePolicy().hasHeightForWidth());
			filePushButton->setSizePolicy(sizePolicy1);
			connect(filePushButton, SIGNAL(clicked()), this, SLOT(selectFileName()));

			horizontalLayout_2->addWidget(filePushButton);


			verticalLayout_2->addLayout(horizontalLayout_2);

			// UniqueCheckBox

			horizontalLayout_3 = new QHBoxLayout();
			horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
			horizontalSpacer = new QSpacerItem(20, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

			horizontalLayout_3->addItem(horizontalSpacer);

			uniqueCheckBox = new QCheckBox(templateOutputGroupBox);
			uniqueCheckBox->setObjectName(QString::fromUtf8("TemplateExportDialog.uniqueCheckBox"));
			connect(uniqueCheckBox, SIGNAL(toggled(bool)), this, SLOT(uniqueToggled(bool)));
			Persistence::Restore(uniqueCheckBox);

			horizontalLayout_3->addWidget(uniqueCheckBox);


			verticalLayout_2->addLayout(horizontalLayout_3);

			// AutosaveCheckBox

			QHBoxLayout* horizontalLayout_3a = new QHBoxLayout();
			horizontalLayout_3a->setObjectName(QString::fromUtf8("horizontalLayout_3"));
			QSpacerItem* horizontalSpacera = new QSpacerItem(20, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

			horizontalLayout_3a->addItem(horizontalSpacera);

			autoSaveCheckBox = new QCheckBox(templateOutputGroupBox);
			autoSaveCheckBox->setObjectName(QString::fromUtf8("TemplateExportDialog.autoSaveCheckBox"));
			//connect(uniqueCheckBox, SIGNAL(toggled(bool)), this, SLOT(uniqueToggled(bool)));
			Persistence::Restore(autoSaveCheckBox);

			horizontalLayout_3a->addWidget(autoSaveCheckBox);


			verticalLayout_2->addLayout(horizontalLayout_3a);


			clipboardRadioButton = new QRadioButton(templateOutputGroupBox);
			clipboardRadioButton->setObjectName(QString::fromUtf8("TemplateExportDialog.clipboardRadioButton"));
			Persistence::Restore(clipboardRadioButton);

			connect(clipboardRadioButton, SIGNAL(toggled(bool)), this, SLOT(fileRadioButtonToggled(bool)));
			
			verticalLayout_2->addWidget(clipboardRadioButton);



			//--

			horizontalLayout_5 = new QHBoxLayout();
			horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
			label_4 = new QLabel(templateOutputGroupBox);
			label_4->setObjectName(QString::fromUtf8("label_4"));
			label_4->setMinimumSize(QSize(10, 10));

			horizontalLayout_5->addWidget(label_4);

			widthSpinBox = new QSpinBox(templateOutputGroupBox);
			widthSpinBox->setRange(0,20000);
			widthSpinBox->setObjectName(QString::fromUtf8("TemplateExportDialog.widthSpinBox"));

			horizontalLayout_5->addWidget(widthSpinBox);

			label_5 = new QLabel(templateOutputGroupBox);
			label_5->setObjectName(QString::fromUtf8("label_5"));

			horizontalLayout_5->addWidget(label_5);

			heightSpinBox = new QSpinBox(templateOutputGroupBox);
			heightSpinBox->setObjectName(QString::fromUtf8("TemplateExportDialog.heightSpinBox"));
			heightSpinBox->setRange(0,20000);
			Persistence::Restore(heightSpinBox); 
			Persistence::Restore(widthSpinBox); 
			connect(heightSpinBox, SIGNAL(valueChanged(int)), this, SLOT(heightChanged(int)));
			connect(widthSpinBox, SIGNAL(valueChanged(int)), this, SLOT(widthChanged(int)));

			
			horizontalLayout_5->addWidget(heightSpinBox);

			line = new QFrame(templateOutputGroupBox);
			line->setObjectName(QString::fromUtf8("line"));
			line->setMinimumSize(QSize(20, 2));
			line->setFrameShape(QFrame::HLine);
			line->setFrameShadow(QFrame::Sunken);

			horizontalLayout_5->addWidget(line);

			lockAspectRatioCheckBox = new QCheckBox(templateOutputGroupBox);
			lockAspectRatioCheckBox->setObjectName(QString::fromUtf8("TemplateExportDialog.lockAspectRatioCheckBox"));
			Persistence::Restore(lockAspectRatioCheckBox);
			//connect(lockAspectRatioCheckBox, SIGNAL(valueChanged()), this, SLOT(lockAspectChanged()));


			horizontalLayout_5->addWidget(lockAspectRatioCheckBox);

			int bw = 40;
			int bh = 20;

			QPushButton* b3 = new QPushButton("/2",templateOutputGroupBox);
			horizontalLayout_5->addWidget(b3);
			b3->setToolTip("Half size.");
			b3->setMinimumSize(QSize(bw, bh));
			b3->setFixedSize(QSize(bw, bh));
			
			QPushButton* b2 = new QPushButton("D",templateOutputGroupBox);
			horizontalLayout_5->addWidget(b2);
			b2->setToolTip("Reset size to OpenGL window size.");
			b2->setMinimumSize(QSize(bw, bh));
			b2->setFixedSize(QSize(bw, bh));
			

			QPushButton* b1 = new QPushButton("*2",templateOutputGroupBox);
			horizontalLayout_5->addWidget(b1);
			b1->setToolTip("Double size.");
			b1->setMinimumSize(QSize(bw, bh));
			b1->setFixedSize(QSize(bw, bh));
			
			

			connect(b3, SIGNAL(clicked()), this, SLOT(halfSize()));
			connect(b2, SIGNAL(clicked()), this, SLOT(defaultSize()));
			connect(b1, SIGNAL(clicked()), this, SLOT(doubleSize()));


			
			

			horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

			horizontalLayout_5->addItem(horizontalSpacer_4);


			verticalLayout_2->addLayout(horizontalLayout_5);

			//--


			verticalLayout_3->addWidget(templateOutputGroupBox);

			postProcessingGroupBox = new QGroupBox(settingstab);
			postProcessingGroupBox->setObjectName(QString::fromUtf8("postProcessingGroupBox"));
			postProcessingGroupBox->setFlat(true);
			verticalLayout_4 = new QVBoxLayout(postProcessingGroupBox);
			verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
			runAfterCheckBox = new QCheckBox(postProcessingGroupBox);
			runAfterCheckBox->setObjectName(QString::fromUtf8("TemplateExportDialog.runAfterCheckBox"));
			Persistence::Restore(runAfterCheckBox);
			
			verticalLayout_4->addWidget(runAfterCheckBox);

			horizontalLayout_4 = new QHBoxLayout();
			horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
			horizontalSpacer_2 = new QSpacerItem(20, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

			horizontalLayout_4->addItem(horizontalSpacer_2);

			afterCommandLineEdit = new QLineEdit(postProcessingGroupBox);
			afterCommandLineEdit->setObjectName(QString::fromUtf8("afterCommandLineEdit"));
				connect(runAfterCheckBox, SIGNAL(toggled(bool)), afterCommandLineEdit, SLOT(setEnabled(bool)));
			
			if (!runAfterCheckBox->isChecked())	afterCommandLineEdit->setEnabled(false);

			horizontalLayout_4->addWidget(afterCommandLineEdit);


			verticalLayout_4->addLayout(horizontalLayout_4);

			runAfterCheckBox->raise();
			afterCommandLineEdit->raise();
			runAfterCheckBox->raise();
			runAfterCheckBox->raise();

			verticalLayout_3->addWidget(postProcessingGroupBox);

			tabWidget->addTab(settingstab, QString());
			advancedTab = new QWidget();
			advancedTab->setObjectName(QString::fromUtf8("advancedTab"));
			verticalLayout_5 = new QVBoxLayout(advancedTab);
			verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));

			modifyTemplateLabel = new QLabel(advancedTab);
			modifyTemplateLabel->setObjectName(QString::fromUtf8("modifyTemplateCheckBox"));

			verticalLayout_5->addWidget(modifyTemplateLabel);

			templateTextEdit = new QTextEdit(advancedTab);
			templateTextEdit->setTabStopWidth(30);
			templateTextEdit->setObjectName(QString::fromUtf8("templateTextEdit"));
			/*XmlHighlighter *highlighter =*/ new XmlHighlighter(templateTextEdit);
			connect(templateTextEdit, SIGNAL(textChanged()), this, SLOT(textChanged()));

			verticalLayout_5->addWidget(templateTextEdit);

			horizontalLayout_6 = new QHBoxLayout();
			horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
			horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

			horizontalLayout_6->addItem(horizontalSpacer_3);

			saveModificationsButton = new QPushButton(advancedTab);
			saveModificationsButton->setObjectName(QString::fromUtf8("pushButton_3"));
			saveModificationsButton->setEnabled(false);
			connect(saveModificationsButton, SIGNAL(clicked()), this, SLOT(saveModifications()));


			horizontalLayout_6->addWidget(saveModificationsButton);

			undoButton = new QPushButton(advancedTab);
			undoButton->setObjectName(QString::fromUtf8("pushButton_4"));
			undoButton->setEnabled(false);
			connect(undoButton, SIGNAL(clicked()), this, SLOT(undo()));


			horizontalLayout_6->addWidget(undoButton);


			verticalLayout_5->addLayout(horizontalLayout_6);

			modifyOutputCheckBox = new QCheckBox(advancedTab);
			modifyOutputCheckBox->setObjectName(QString::fromUtf8("TemplateExportDialog.modifyOutputCheckBox"));

			Persistence::Restore(modifyOutputCheckBox);

			verticalLayout_5->addWidget(modifyOutputCheckBox);

			tabWidget->addTab(advancedTab, QString());

			verticalLayout->addWidget(tabWidget);

			buttonBox = new QDialogButtonBox(this);
			buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
			buttonBox->setOrientation(Qt::Horizontal);
			buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

			verticalLayout->addWidget(buttonBox);


			retranslateUi();
			QObject::connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
			QObject::connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

			tabWidget->setCurrentIndex(0);

			connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));

			uniqueToggled(true);

	//		connect(runAfterLineEdit, SIGNAL(textChanged(const QString &)),this, SLOT(updateUniqueFileName(const QString &)));

			//QMetaObject::connectSlotsByName(this);
		} // setupUi

		void TemplateExportDialog::setTemplatePath(QString templatePath) {

			if (Persistence::Contains("TemplateExportDialog::TemplatePath")) {
				INFO("Custom template path chosen: " + templatePath);
				templatePath = Persistence::Get("TemplateExportDialog::TemplatePath").toString();
			}

			// Scan render templates...
			QDir dir(templatePath);
			QStringList filters;
			filters << "*.rendertemplate";
			dir.setNameFilters(filters);
			bool failed = false;
			if (!dir.exists()) {
				WARNING("Unable to locate: "+dir.absolutePath());
				failed = true;
			} else {
				QStringList sl = dir.entryList();
				templateComboBox->clear();
				for (int i = 0; i < sl.size(); i++) {
					templateComboBox->insertItem(10000, sl[i].remove(".rendertemplate", Qt::CaseInsensitive), 
						dir.absoluteFilePath(sl[i]));
				}

				if (sl.size() == 0) {
					failed = true;
				}
			}
			if (failed) {
				templateTextEdit->setText("");
				modifiedTemplate = "";
				descriptionTextBrowser->setText("Unable to read templates from: " + templatePath);
				primitivesTableWidget->clear();
			}

			Persistence::Restore(templateComboBox);



		}

		void TemplateExportDialog::selectFileName() {
			QString filter = currentTemplate.getDefaultExtension() + ";;All Files (*.*)";
			
			QString fileName = QFileDialog::getSaveFileName(this, tr("Save As"), fileNameLineEdit->text(), filter);
			if (fileName.isEmpty()) {
				return;
			}
			fileNameLineEdit->setText(fileName);
			updateUniqueFileName("");
		}

		void TemplateExportDialog::templateChanged(const QString& s) {
			int id = templateComboBox->currentIndex();

			if (id<0) return;

			

			if (!s.isEmpty() && saveModificationsButton->isEnabled()) {
				// TODO:
				WARNING("You have lost your changes...");
				modifiedTemplate = "";
			}

			QVariant q = templateComboBox->itemData(id);

			QFile file(q.toString());
			
			primitivesTableWidget->setColumnCount(1);
			primitivesTableWidget->horizontalHeader()->hide();
				
			

			try {
				// If the template has been modified, the modifications
				// are stored in modifiedTemplate.
				if (modifiedTemplate.isEmpty()) {
					Template t(file);
					currentTemplate = t;
				} else {
					Template t(modifiedTemplate);
					modifiedTemplate = "";
					currentTemplate = t;
				}


				templateTextEdit->setText(
					currentTemplate.getFullText());

				if (!currentTemplate.getRunAfter().isEmpty()) {
				afterCommandLineEdit->setText(currentTemplate.getRunAfter());
				}

				QString runAfterText = Persistence::Get("TemplateExportDialog.runAfter." + templateComboBox->currentText()).toString();
				if (!runAfterText.isEmpty()) {
					afterCommandLineEdit->setText(runAfterText);
				}
					
				undoButton->setEnabled(false);
				saveModificationsButton->setEnabled(false);

				QString html = currentTemplate.getDescription();
				html = html.replace("\n", "<br>");
				descriptionTextBrowser->setText(
					"<b>Name:</b> "+ currentTemplate.getName() + "<br>\r\n" + "<b>File type:</b> " + currentTemplate.getDefaultExtension() + "\r\n<br>" + "\r\n" +
					html);

				int count = 0;
				foreach (QString p, primitives) {
					if (!currentTemplate.getPrimitives().contains(p)) {
						QTableWidgetItem* item = new QTableWidgetItem( p);
						item->setBackground(QBrush(Qt::red));
						
						primitivesTableWidget->setRowCount(count+1);

						primitivesTableWidget->setItem(count, 0, item);
						count++;						
					}
				}
				primitivesTableWidget->setRowCount(count+currentTemplate.getPrimitives().count());


				QMapIterator<QString, TemplatePrimitive> i(currentTemplate.getPrimitives());
				while (i.hasNext()) {
					i.next();
					QTableWidgetItem* item = new QTableWidgetItem( i.key());
					if (primitives.contains(i.key())) {
						item->setBackground(QBrush(Qt::green));
					}
					primitivesTableWidget->setItem(count, 0, item);

					count++;
				}
				
				
				changeFileNameExtension(currentTemplate.getDefaultExtension());
			} catch (Exception& e) {
				primitivesTableWidget->setRowCount(0);
				
				WARNING(e.getMessage());
				templateTextEdit->setText(modifiedTemplate);	
				descriptionTextBrowser->setText(e.getMessage());
		
				currentTemplate = Template();
			}

			
		}

		void TemplateExportDialog::tabChanged(int i) {
			if ((i == 0) && saveModificationsButton->isEnabled()) {
				modifiedTemplate = templateTextEdit->toPlainText();
				templateChanged("");
			}
		}


		void TemplateExportDialog::changeFileNameExtension(QString extension) {

			QRegExp rx("\\(\\*\\.(.*)\\)"); // extract stuff inside brackets
			int pos = 0;

			QString realExtension = "*.unknown";
			if (rx.indexIn(extension, pos) != -1) {
				realExtension = rx.cap(1);
			}


			QString stripped = fileNameLineEdit->text().section(".",0,-2); // find everything until extension.
			fileNameLineEdit->setText(stripped + "." + realExtension);
		}


		void TemplateExportDialog::uniqueToggled(bool) {
			updateUniqueFileName("");
		}

		void TemplateExportDialog::updateUniqueFileName(const QString &) {
			uniqueFileName = "";
			QString uname = "";
			QString file = fileNameLineEdit->text();
			QFileInfo fi(file);
			bool error = false;
			if (!fi.absoluteDir().exists()) {
				uname = "dir does not exist";
				error = true;
				QPalette p = fileNameLineEdit->palette();
				p.setColor(QPalette::Base, QColor(255,70,70));
				fileNameLineEdit->setPalette(p);
			} else {
				fileNameLineEdit->setPalette(QApplication::palette());
			}
				
			if (uniqueCheckBox->isChecked()) {
				if (!error) {

					QString stripped = fileNameLineEdit->text().section(".",0,-2); // find everything until extension.
					QString extension = fileNameLineEdit->text().section(".",-1,-1); 

					QString lastNumber = stripped.section("-", -1, -1);
					bool succes = false;
					int number = lastNumber.toInt(&succes);
					if (!succes) number = 2;
					if (succes) {
						// The filename already had a number extension.
						stripped = stripped.section("-", 0, -2);
					}

					QString testName = fileNameLineEdit->text();
					while (QFile(testName).exists()) {
						testName = stripped + "-" + QString::number(number++) + "." + extension;
					}
					uname = testName;
					uniqueFileName = uname;
				}
				
				uniqueCheckBox->setText(QString("Add unique ID to filename (%1)").arg(uname));
				if (autoSaveCheckBox) autoSaveCheckBox->setText(QString("Autosave Eisenscript (%1)").arg(uname.section(".",0,-2)+".es"));
			} else {
				
				uniqueCheckBox->setText("Add unique ID to filename");
				if (autoSaveCheckBox) autoSaveCheckBox->setText(QString("Autosave Eisenscript (%1)").arg(
					 fileNameLineEdit->text().section(".",0,-2)+".es"
					));
			}
		}

		void TemplateExportDialog::retranslateUi()
		{
			setWindowTitle(QApplication::translate("Dialog", "Template Export", 0, QApplication::UnicodeUTF8));
			label->setText(QApplication::translate("Dialog", "Template:", 0, QApplication::UnicodeUTF8));
			templatePathButton->setText(QApplication::translate("Dialog", "Template Path...", 0, QApplication::UnicodeUTF8));
			descriptionLabel->setText(QApplication::translate("Dialog", "Description", 0, QApplication::UnicodeUTF8));
			descriptionTextBrowser->setHtml(QApplication::translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
				"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
				"p, li { white-space: pre-wrap; }\n"
				"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
				"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">This template was created by ....</span></p></body></html>", 0, QApplication::UnicodeUTF8));
			primitivesLabel->setText(QApplication::translate("Dialog", "Primitives in Template", 0, QApplication::UnicodeUTF8));
			templateOutputGroupBox->setTitle(QApplication::translate("Dialog", "Template Output", 0, QApplication::UnicodeUTF8));
			fileRadioButton->setText(QApplication::translate("Dialog", "File:", 0, QApplication::UnicodeUTF8));
			filePushButton->setText(QApplication::translate("Dialog", "File...", 0, QApplication::UnicodeUTF8));
			//uniqueCheckBox->setText(QApplication::translate("Dialog", "Add unique ID to filename", 0, QApplication::UnicodeUTF8));
			clipboardRadioButton->setText(QApplication::translate("Dialog", "Clipboard", 0, QApplication::UnicodeUTF8));
			postProcessingGroupBox->setTitle(QApplication::translate("Dialog", "Post Processing", 0, QApplication::UnicodeUTF8));
			runAfterCheckBox->setText(QApplication::translate("Dialog", "Run the following command after export:", 0, QApplication::UnicodeUTF8));
			afterCommandLineEdit->setText(QApplication::translate("Dialog", "", 0, QApplication::UnicodeUTF8));
			tabWidget->setTabText(tabWidget->indexOf(settingstab), QApplication::translate("Dialog", "Settings", 0, QApplication::UnicodeUTF8));
			label_4->setText(QApplication::translate("Dialog", "Width:", 0, QApplication::UnicodeUTF8));
			label_5->setText(QApplication::translate("Dialog", "Height:", 0, QApplication::UnicodeUTF8));
			lockAspectRatioCheckBox->setText(QApplication::translate("Dialog", "Lock aspect ratio (Current = 1.23)", 0, QApplication::UnicodeUTF8));
			modifyTemplateLabel->setText(QApplication::translate("Dialog", "Modify template before applying", 0, QApplication::UnicodeUTF8));
			saveModificationsButton->setText(QApplication::translate("Dialog", "Save Modifications", 0, QApplication::UnicodeUTF8));
			undoButton->setText(QApplication::translate("Dialog", "Revert (Undo Changed)", 0, QApplication::UnicodeUTF8));
			modifyOutputCheckBox->setText(QApplication::translate("Dialog", "Modify output before saving (spawns edit window when pressing OK)", 0, QApplication::UnicodeUTF8));
			tabWidget->setTabText(tabWidget->indexOf(advancedTab), QApplication::translate("Dialog", "Modify", 0, QApplication::UnicodeUTF8));

		} // retranslateUi

		void TemplateExportDialog::fileRadioButtonToggled(bool) {
			if (fileRadioButton->isChecked()) {
				fileNameLineEdit->setEnabled(true);
				uniqueCheckBox->setEnabled(true);
				autoSaveCheckBox->setEnabled(true);
				filePushButton->setEnabled(true);
			} else {
				fileNameLineEdit->setEnabled(false);
				uniqueCheckBox->setEnabled(false);
				filePushButton->setEnabled(false);
				autoSaveCheckBox->setEnabled(false);
			}
		}

		void TemplateExportDialog::undo() {
			
			templateTextEdit->setText(
			currentTemplate.getFullText());

			undoButton->setEnabled(false);
			saveModificationsButton->setEnabled(false);
		}

		void TemplateExportDialog::saveModifications() {
			int id = templateComboBox->currentIndex();
			if (id<0) return;
			QVariant q = templateComboBox->itemData(id);

			QFile file(q.toString());

			if (!QFileInfo(file).exists()) {
				WARNING("Could not find file: " + QFileInfo(file).absoluteFilePath());
				return;
			}

			if (QMessageBox::Ok != QMessageBox::warning (this, "Overwrite file?", 
				QString("Overwrite template:\r\n%1").arg(QFileInfo(file).absoluteFilePath()), QMessageBox::Ok | QMessageBox::Cancel, QMessageBox::Cancel))
			{
				return;
			}

			if (file.open(QIODevice::WriteOnly)) {
				file.write(templateTextEdit->toPlainText().toLatin1());
				file.flush();
				file.close();
				INFO("Wrote template to file: " + QFileInfo(file).absoluteFilePath());	
			} else {
				WARNING("Could not write to file: " + QFileInfo(file).absoluteFilePath());
				return;
			}

			modifiedTemplate = templateTextEdit->toPlainText();
			templateChanged("");

			undoButton->setEnabled(false);
			saveModificationsButton->setEnabled(false);


		}

		void TemplateExportDialog::textChanged() {
			undoButton->setEnabled(true);
			saveModificationsButton->setEnabled(true);

		}

		void TemplateExportDialog::changeTemplatePath() {
			QString fileName = QFileDialog::getExistingDirectory(
				this, tr("Choose Template folder"),
                 "",
                 QFileDialog::ShowDirsOnly      | QFileDialog::DontResolveSymlinks);
			if (fileName.isEmpty()) {
				return;
			}
			Persistence::Put("TemplateExportDialog::TemplatePath", fileName);
			setTemplatePath("");
		}

		void TemplateExportDialog::accept() {
			
			QString fileName = "";
			if (fileRadioButton->isChecked()) {
				
				if (uniqueCheckBox->isChecked()) {
					fileName = uniqueFileName;
				} else {
					fileName = fileNameLineEdit->text();
				}

				
				if (QFileInfo(fileName).exists()) {
					if (QMessageBox::Ok != QMessageBox::warning (this, "File exists!", 
						QString("Overwrite file:\r\n%1").arg(QFileInfo(fileName).absoluteFilePath()), QMessageBox::Ok | QMessageBox::Cancel, QMessageBox::Cancel))
					{
						return;
					}
				}

				if (autoSaveCheckBox->isChecked()) {

					QString autoSaveName = fileName.section(".",0,-2) + ".es";
					
					if (QFileInfo(autoSaveName).exists()) {
						if (QMessageBox::Ok != QMessageBox::warning (this, "File exists!", 
							QString("Overwrite file:\r\n%1").arg(QFileInfo(autoSaveName).absoluteFilePath()), QMessageBox::Ok | QMessageBox::Cancel, QMessageBox::Cancel))
						{
							return;
						}
					}

					INFO("Autosaving Eisenscript as: " + QFileInfo(autoSaveName).absoluteFilePath());
					QString script = mainWindow->getScriptWithSettings(fileName);

					QFile file(autoSaveName);
					if (!file.open(QFile::WriteOnly | QFile::Text)) {
						QMessageBox::warning(this, tr("Structure Synth"),
							tr("Cannot write file %1:\n%2.")
							.arg(autoSaveName)
							.arg(file.errorString()));
						return;
					}

					QTextStream out(&file);
					QApplication::setOverrideCursor(Qt::WaitCursor);
					out << script;
					QApplication::restoreOverrideCursor();
					INFO("Eisenscript saved.");
				}

				mainWindow->templateRender(fileName, &currentTemplate, "", widthSpinBox->value(), heightSpinBox->value(), modifyOutputCheckBox->isChecked());

			} else {
				// Save to clipboard.
				INFO("Rendering to clipboard...");
				mainWindow->templateRender("", &currentTemplate, "", widthSpinBox->value(), heightSpinBox->value(), modifyOutputCheckBox->isChecked());
			}

			if (runAfterCheckBox->isChecked()) {
				QString cmd = afterCommandLineEdit->text();
				cmd = cmd.replace("$FILE", QFileInfo(fileName).absoluteFilePath());

				QStringList env = QProcess::systemEnvironment();
				foreach (QString es, env) {
					QStringList l = es.split("=");
					if (l.count() == 2) {
						cmd = cmd.replace("%"+l[0]+"%", l[1]);
					}
				}


				bool inQuote = false;
				QStringList args;
				QString command;
				int counter = 0;
				QString buffer;
				INFO("Post-processed text: " + cmd);
				for (int i = 0; i < cmd.size(); ++i) {
					 if (cmd.at(i) == QLatin1Char('"')) {
						 inQuote = !inQuote;
						 continue;
					 }
					 
					 if (cmd.at(i) == ' ' && !inQuote) {
						 if (counter++ == 0) {
							 command = buffer;
						 } else {
							 args.append(buffer);
						 }
						 buffer = "";
						 continue;
					 }

					 buffer += cmd.at(i);	
				}
				if (!buffer.isEmpty())  args.append(buffer);

				INFO("Command: " + command);
				INFO("Args: " + args.join(" "));
				QString dir = QFileInfo(command).absolutePath();
				INFO("Working Directory: " + dir);
   
				if (QProcess::startDetached(command, args, dir)) {
					INFO("Started process");
				} else {
					WARNING("Failed to start process");
				}
			}

			// Persist changes to UI...
			Persistence::Store(fileNameLineEdit);
			Persistence::Store(lockAspectRatioCheckBox); // checkb
			Persistence::Store(widthSpinBox); // spin
			Persistence::Store(heightSpinBox); // spin
			Persistence::Store(uniqueCheckBox); // checkb
			Persistence::Store(autoSaveCheckBox); 
			Persistence::Store(clipboardRadioButton);
			Persistence::Store(templateComboBox);
			Persistence::Store(runAfterCheckBox);
			Persistence::Store(modifyOutputCheckBox);

			Persistence::Put("TemplateExportDialog.runAfter." + templateComboBox->currentText()
				,afterCommandLineEdit->text());
			
			

			QDialog::accept();
		};

		void TemplateExportDialog::reject() {
			QDialog::reject();
		};
			

	};


}

