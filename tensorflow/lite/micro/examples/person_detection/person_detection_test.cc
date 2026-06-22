/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* [빈칸] 모델 header 파일(tensorflow/lite/micro/models/person_detect_model_data.h)을 include하세요 */
#include "?"

/* Include test data */
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/person_image_data.h"

/* Unit test header를 include합니다 */
#include "tensorflow/lite/micro/testing/micro_test.h"

/* TEST 시작 매크로 */
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  /* [빈칸] person_detect_model_data.h에 선언된 모델 배열(g_person_detect_model_data)을 로드하세요. */
  const tflite::Model* model = ::tflite::GetModel(?);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.

  /* [빈칸] MicroMutableOpResolver 인스턴스를 만들고, 모델 수행에 필요한 아래 5개의 operation을 추가합니다 */
  /* AddAveragePool2D, AddConv2D, AddDepthwiseConv2D, AddReshape, AddSoftmax */
  tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.?;
  micro_op_resolver.?;
  micro_op_resolver.?;
  micro_op_resolver.?;
  micro_op_resolver.?;

  // Create an area of memory to use for input, output, and intermediate arrays.
  /* [빈칸] 136 * 1024 크기의 int 배열을 만드세요 (이름: tensor_arena). */
  constexpr int tensor_arena_size = ?;
  uint8_t ?[tensor_arena_size];

  // Build an interpreter to run the model with.
  /* [빈칸] interpreter는 모델, 연산, 텐서아레나/사이즈, 로거를 인자로 받아 build합니다. */
  /* [힌트] micro_op_resolver, tensor_arena, tensor_arena_size, error_reporter */
  tflite::MicroInterpreter interpreter(model, ?, ?, ?, ?);

  //tensor 메모리를 할당
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kNumRows, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kNumCols, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kNumChannels, input->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, input->type);

  TFLITE_DCHECK_EQ(input->bytes, static_cast<size_t>(g_person_image_data_size));

///////////////////////////////// 사람(person) 이미지 추론 /////////////////////////////////

  // Copy an image with a person into the memory area used for the input.
  memcpy(input->data.int8, g_person_image_data, input->bytes);

  // Run the model on this input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and type.
  TfLiteTensor* output = interpreter.output(0);
 
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output->type);

  int8_t person_score = output->data.int8[kPersonIndex];
  int8_t no_person_score = output->data.int8[kNotAPersonIndex];
  TF_LITE_REPORT_ERROR(&micro_error_reporter,
                       "person data.  person score: %d, no person score: %d\n",
                       person_score, no_person_score);

  // Make sure that the expected "Person" score is higher than the other class.
  TF_LITE_MICRO_EXPECT_GT(person_score, no_person_score);


///////////////////////////////// 기린(no_person) 이미지 추론 /////////////////////////////////

  /* [빈칸] 입력 텐서(input->data.int8)에 input->bytes 크기의 g_no_person_image_data를 copy하세요. */
  /* [힌트] memcpy 함수의 입력인자는 (target, reference, size) 순서입니다. */
  memcpy(?, ?, ?);

  // Run the model on this "No Person" input.
  /* [빈칸] interpreter의 Invoke()를 호출하여 모델을 실행하세요. */
  invoke_status = ?.?();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and type.
  output = interpreter.output(0);

  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output->type);

  /* [빈칸] person_score를 output->data.int8[kPersonIndex]에서 받아오세요. */
  person_score = ?->?.?[?];
  /* [빈칸] no_person_score를 output->data.int8[kNotAPersonIndex]에서 받아오세요. */
  no_person_score = ?->?.?[?];
  TF_LITE_REPORT_ERROR(
      &micro_error_reporter,
      "no person data.  person score: %d, no person score: %d\n", person_score,
      no_person_score);
  
  // Make sure that the expected "No Person" score is higher.
  /* [빈칸] no_person_score가 person_score보다 높은지 확인하세요. */
  TF_LITE_MICRO_EXPECT_GT(?, ?);

  TF_LITE_REPORT_ERROR(&micro_error_reporter, "Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
