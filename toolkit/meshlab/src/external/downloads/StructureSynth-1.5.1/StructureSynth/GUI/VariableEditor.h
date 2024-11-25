#pragma once

#include <QString>
#include <QVector>
#include <QWidget>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QHBoxLayout>

#include "../Parser/Preprocessor.h"

/// Classes for the GUI Editor for the preprocessor constant variables.
/// E.g. the line: #define angle 45 (float:0.0-360.0)
///	will make a simple editor widget appear.
namespace StructureSynth {
	namespace GUI {
	
		class VariableWidget; // Forward decl...

		/// The Variable Editor window.
		class VariableEditor : public QWidget {
		public:
			VariableEditor(QWidget* parent);

			QString updateFromPreprocessor(Parser::Preprocessor* pp, QString in, bool* showGUI);

		private:
			
			QSpacerItem* spacer;
			QVector<VariableWidget*> variables;
			QVBoxLayout* layout;
		};

		// A helper class (combined float slider+spinner)
		class ComboSlider : public QWidget {
		Q_OBJECT
		public:
			ComboSlider(QWidget* parent, double defaultValue, double minimum, double maximum) 
				: QWidget(parent), defaultValue(defaultValue), minimum(minimum), maximum(maximum){
				setLayout(new QHBoxLayout());
				slider = new QSlider(Qt::Horizontal,this);
				slider->setRange(0,1000);
				double val = (defaultValue-minimum)/(maximum-minimum);
				slider->setValue(val*1000);
				spinner = new QDoubleSpinBox(this);
				spinner->setMaximum(maximum);
				spinner->setMinimum(minimum);
				spinner->setValue(defaultValue);
				layout()->addWidget(slider);
				layout()->addWidget(spinner);
				connect(spinner, SIGNAL(valueChanged(double)), this, SLOT(spinnerChanged(double)));
				connect(slider, SIGNAL(valueChanged(int)), this, SLOT(sliderChanged(int)));
			}

			double getValue() { return spinner->value(); }
		
		protected slots:
			void spinnerChanged(double) {
				double val = (spinner->value()-minimum)/(maximum-minimum);
				slider->setValue(val*1000);
			}

			void sliderChanged(int) {
				double val = (slider->value()/1000.0)*(maximum-minimum)+minimum;
				spinner->setValue(val);
			}

		private:
			
			QSlider* slider;
			QDoubleSpinBox* spinner;
			double defaultValue;
			double minimum;
			double maximum;
		};

		// A helper class (combined int slider+spinner)
		class IntComboSlider : public QWidget {
		Q_OBJECT
		public:
			IntComboSlider(QWidget* parent, int defaultValue, int minimum, int maximum) 
				: QWidget(parent), defaultValue(defaultValue), minimum(minimum), maximum(maximum){
				setLayout(new QHBoxLayout());
				slider = new QSlider(Qt::Horizontal,this);
				slider->setRange(minimum,maximum);
				slider->setValue(defaultValue);
				spinner = new QSpinBox(this);
				spinner->setMaximum(maximum);
				spinner->setMinimum(minimum);
				spinner->setValue(defaultValue);
				layout()->addWidget(slider);
				layout()->addWidget(spinner);
				connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(spinnerChanged(int)));
				connect(slider, SIGNAL(valueChanged(int)), this, SLOT(sliderChanged(int)));
			}

			int getValue() { return spinner->value(); }
		
		protected slots:
			void spinnerChanged(int) {
				int val = spinner->value();
				slider->setValue(val);
			}

			void sliderChanged(int) {
				double val = slider->value();
				spinner->setValue(val);
			}

		private:
			
			QSlider* slider;
			QSpinBox* spinner;
			int defaultValue;
			int minimum;
			int maximum;
		};

	}
}

