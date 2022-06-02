
.. _program_listing_file_include_test.h:

Program Listing for File test.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_include_test.h>` (``include/test.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   class QTstyle_Test
   {
     public:
    
   
       enum TEnum { 
                    TVal1, 
                    TVal2, 
                    TVal3  
                  } 
   
            *enumPtr, 
   
            enumVar;  
       
   
       QTstyle_Test();
    
   
      ~QTstyle_Test();
       
   
       int testMe(int a,const char *s);
          
   
       virtual void testMeToo(char c1,char c2) = 0;
      
   
       int publicVar;
          
   
       int (*handler)(int a,int b);
   };
