#include "VariableEditor.h"

#include <QPushButton>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>

#include "../../SyntopiaCore/Logging/ListWidgetLogger.h"

using namespace SyntopiaCore::Logging;


namespace StructureSynth {
	namespace GUI {

	
		/// Widget editor base class.
		class VariableWidget : public QWidget {
		public:
			VariableWidget(QWidget* parent, QString name) : QWidget(parent), name(name) {};

			virtual QString getValueAsText() { return ""; };
			QString getName() const { return name; };
			bool isUpdated() const { return updated; };
			void setUpdated(bool value) { updated = value; };

		private:
			QString name;
			bool updated;
			QWidget* widget;
		};


	

		/// A widget editor for a float variable.
		class FloatWidget : public VariableWidget {
		public:
			/// FloatVariable constructor.
			FloatWidget(QWidget* parent, QString name, double defaultValue, double min, double max) 
				: VariableWidget(parent, name)  {
					QHBoxLayout* l = new QHBoxLayout(this);
					l->setSpacing(2);
					setContentsMargins (0,0,0,0);
					QLabel* label = new QLabel(this);
					label->setText(name);
					l->addWidget(label);
					comboSlider = new ComboSlider(parent, defaultValue, min, max);
					l->addWidget(comboSlider);
			};

			virtual QString getValueAsText() { return QString::number(comboSlider->getValue()); };

		private:
			ComboSlider* comboSlider;
		};


		class IntWidget : public VariableWidget {
		public:
			/// FloatVariable constructor.
			IntWidget(QWidget* parent, QString name, int defaultValue, int min, int max) 
				: VariableWidget(parent, name)  {
					QHBoxLayout* l = new QHBoxLayout(this);
					l->setSpacing(2);
					setContentsMargins (0,0,0,0);
					QLabel* label = new QLabel(this);
					label->setText(name);
					l->addWidget(label);
					comboSlider = new IntComboSlider(parent, defaultValue, min, max);
					l->addWidget(comboSlider);
			};

			virtual QString getValueAsText() { return QString::number(comboSlider->getValue()); };

		private:
			IntComboSlider* comboSlider;
		};


		VariableEditor::VariableEditor(QWidget* parent) : QWidget(parent) {
			layout = new QVBoxLayout(this);
			layout->setSpacing(1);
			spacer=0;
		};	

	
		QString VariableEditor::updateFromPreprocessor(Parser::Preprocessor* pp, QString in, bool* showGUI) {
			if (spacer) {
				layout->removeItem(spacer);
				delete(spacer);
				spacer = 0;
			}
			QVector<Parser::GuiParameter*> ps = pp->getParameters();
			QMap<QString, QString> substitutions;


			for (int i = 0; i < variables.count(); i++) {
				QString name = variables[i]->getName();
				variables[i]->setUpdated(false);
			}

			for (int i = 0; i < ps.count(); i++) {
				bool found = false;
				for (int j = 0; j < variables.count(); j++) {
					QString name = variables[j]->getName();
					if (name == ps[i]->getName()) {
						substitutions[name] = variables[j]->getValueAsText();
						found = true;
						variables[j]->setUpdated(true);
						//INFO("Found existing: " + variables[j]->getName() + QString(" value: %1").arg(variables[j]->getValueAsText()));
					}
				}

				if (!found) {
					if (dynamic_cast<Parser::FloatParameter*>(ps[i])) {
						Parser::FloatParameter* fp = dynamic_cast<Parser::FloatParameter*>(ps[i]);
						QString name = fp->getName();
						FloatWidget* fw = new FloatWidget(this, name, fp->getDefaultValue(), fp->getFrom(), fp->getTo());
						variables.append(fw);
						fw->setUpdated(true);
						layout->addWidget(fw);
						substitutions[name] = fw->getValueAsText();
					}

					if (dynamic_cast<Parser::IntParameter*>(ps[i])) {
						Parser::IntParameter* ip = dynamic_cast<Parser::IntParameter*>(ps[i]);
						QString name = ip->getName();
						IntWidget* iw = new IntWidget(this, name, ip->getDefaultValue(), ip->getFrom(), ip->getTo());
						variables.append(iw);
						iw->setUpdated(true);
						layout->addWidget(iw);
						substitutions[name] = iw->getValueAsText();
					}
				}
			}

			for (int i = 0; i < variables.count(); ) {
				if (!variables[i]->isUpdated()) {
					//INFO("Deleting : " + variables[i]->getName());
					delete(variables[i]);
					variables.remove(i);
					i = 0;

				} else {
					i++;
				}
			}

			if (showGUI) (*showGUI) = (variables.count() != 0);

			QMap<QString, QString>::const_iterator it2 = substitutions.constBegin();
			int subst = 0;
			while (it2 != substitutions.constEnd()) {
				if (subst>100) {
					WARNING("More than 100 recursive preprocessor substitutions... breaking.");
					break;
				}
				if (in.contains(it2.key())) {
					//INFO("Replacing: " + it2.key() + " with " + it2.value());
					in.replace(it2.key(), it2.value());

					it2 = substitutions.constBegin();
					subst++;
				} else {
					it2++;
				}
			}
			spacer = new QSpacerItem(1,1, QSizePolicy::Minimum,QSizePolicy::Expanding);
			layout->addItem(spacer);
		
			return in;
		}

			

	}
}

