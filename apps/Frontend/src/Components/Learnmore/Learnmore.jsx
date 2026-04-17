import React from "react";
import"./Learnmore.css"

const StepsSection = () => {
  return (
    <div className="steps-wrapper text-center py-5">
      <div className="container">

        <h1 className="main-title mb-3">
          How To Use Our Advanced AI– Prediction Tool
        </h1>

        <p className="description mx-auto mb-5">
  <span className="line line1">
    Heart Diseases Are Very Dangerous So If You Want To Take Care Of Your
  </span>
  <span className="line line2">
    Heart Follow The Following Steps To Make Sure That You Get The
  </span>
  <span className="line line3">
    Most Benefit From Our Site.
  </span>
</p>

        <div className="row justify-content-center g-4">

  {/* Card 1 */}
  <div className="col-md-3 d-flex justify-content-center">
    <div className="step-card step-card-1">
      <div className="step-circle">1</div>
      <h5>First Step</h5>
      <h6>Check Up</h6>
      <p>
        You Should Check Your Heart Care Always To Take Care Of Your
        Health And You Should Do That With The Right Way
      </p>
    </div>
  </div>

  {/* Card 2 */}
  <div className="col-md-6 d-flex justify-content-center">
    <div className="step-card step-card-2">
      <div className="step-circle">2</div>
      <h5>Second Step</h5>
      <h6>The Labs</h6>
      <p>
        You Should Go To A Specialized And Trusted Labs To Check
        Your Heart Care And It Is Suggested Doing The Check Up
        And Examination Under Medical Supervision
      </p>
    </div>
  </div>

  {/* Card 3 */}
  <div className="col-md-3 d-flex justify-content-center">
    <div className="step-card step-card-3">
      <div className="step-circle">3</div>
      <h5>Third Step</h5>
      <h6>The Medical Report</h6>
      <p>
        After Finish Your Tests And Examinations In The Lap,
        The Lap Will Send The Report File To Our System,
        Then You Can Start Prediction
      </p>
    </div>
  </div>

</div>
      </div>
    </div>
  );
};

export default StepsSection;