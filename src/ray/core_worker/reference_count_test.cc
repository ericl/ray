#include <vector>

#include "gtest/gtest.h"
#include "ray/common/ray_object.h"
#include "ray/core_worker/reference_count.h"
#include "ray/core_worker/store_provider/memory_store/memory_store.h"

namespace ray {

class ReferenceCountTest : public ::testing::Test {
 protected:
  std::unique_ptr<ReferenceCounter> rc;
  virtual void SetUp() { rc = std::unique_ptr<ReferenceCounter>(new ReferenceCounter); }

  virtual void TearDown() {}
};

// Tests basic incrementing/decrementing of direct reference counts. An entry should only
// be removed once its reference count reaches zero.
TEST_F(ReferenceCountTest, TestBasic) {
  ObjectID id = ObjectID::FromRandom();
  rc->AddReference(id);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 1);
  rc->AddReference(id);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 1);
  rc->AddReference(id);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 1);
  rc->RemoveReference(id);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 1);
  rc->RemoveReference(id);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 1);
  rc->RemoveReference(id);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 0);
}

// Tests the basic logic for dependencies - when an ObjectID with dependencies
// goes out of scope (i.e., reference count reaches zero), all of its dependencies
// should have their reference count decremented and be removed if it reaches zero.
TEST_F(ReferenceCountTest, TestDependencies) {
  ObjectID id1 = ObjectID::FromRandom();
  ObjectID id2 = ObjectID::FromRandom();
  ObjectID id3 = ObjectID::FromRandom();

  std::shared_ptr<std::vector<ObjectID>> deps = std::make_shared<std::vector<ObjectID>>();
  deps->push_back(id2);
  deps->push_back(id3);
  rc->SetDependencies(id1, deps);

  rc->AddReference(id1);
  rc->AddReference(id1);
  rc->AddReference(id3);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 3);

  rc->RemoveReference(id1);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 3);
  rc->RemoveReference(id1);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 1);

  rc->RemoveReference(id3);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 0);
}

// Tests the case where two entries share the same set of dependencies. When one
// entry goes out of scope, it should decrease the reference count for the dependencies
// but they should still be nonzero until the second entry goes out of scope and all
// direct dependencies to the dependencies are removed.
TEST_F(ReferenceCountTest, TestSharedDependencies) {
  ObjectID id1 = ObjectID::FromRandom();
  ObjectID id2 = ObjectID::FromRandom();
  ObjectID id3 = ObjectID::FromRandom();
  ObjectID id4 = ObjectID::FromRandom();

  std::shared_ptr<std::vector<ObjectID>> deps = std::make_shared<std::vector<ObjectID>>();
  deps->push_back(id3);
  deps->push_back(id4);
  rc->SetDependencies(id1, deps);
  rc->SetDependencies(id2, deps);

  rc->AddReference(id1);
  rc->AddReference(id2);
  rc->AddReference(id4);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 4);

  rc->RemoveReference(id1);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 3);
  rc->RemoveReference(id2);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 1);

  rc->RemoveReference(id4);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 0);
}

// Tests the case when an entry has a dependency that itself has a
// dependency. In this case, when the first entry goes out of scope
// it should decrease the reference count for its dependency, causing
// that entry to go out of scope and decrease its dependencies' reference counts.
TEST_F(ReferenceCountTest, TestRecursiveDependencies) {
  ObjectID id1 = ObjectID::FromRandom();
  ObjectID id2 = ObjectID::FromRandom();
  ObjectID id3 = ObjectID::FromRandom();
  ObjectID id4 = ObjectID::FromRandom();

  std::shared_ptr<std::vector<ObjectID>> deps1 =
      std::make_shared<std::vector<ObjectID>>();
  deps1->push_back(id2);
  rc->SetDependencies(id1, deps1);

  std::shared_ptr<std::vector<ObjectID>> deps2 =
      std::make_shared<std::vector<ObjectID>>();
  deps2->push_back(id3);
  deps2->push_back(id4);
  rc->SetDependencies(id2, deps2);

  rc->AddReference(id1);
  rc->AddReference(id2);
  rc->AddReference(id4);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 4);

  rc->RemoveReference(id2);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 4);
  rc->RemoveReference(id1);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 1);

  rc->RemoveReference(id4);
  ASSERT_EQ(rc->NumObjectIDsInScope(), 0);
}

// Tests that the ref counts are properly integrated into the local
// object memory store.
TEST(MemoryStoreIntegrationTest, TestSimple) {
  ObjectID id1 = ObjectID::FromRandom();
  ObjectID id2 = ObjectID::FromRandom();
  uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  RayObject buffer(std::make_shared<LocalMemoryBuffer>(data, sizeof(data)), nullptr);

  auto rc = std::shared_ptr<ReferenceCounter>(new ReferenceCounter());
  CoreWorkerMemoryStore store(rc);

  // Tests adding an object for an existing reference.
  rc->AddReference(id1);
  rc->AddReference(id1);
  RAY_CHECK_OK(store.Put(id1, buffer));
  ASSERT_EQ(store.Size(), 1);
  rc->RemoveReference(id1);
  ASSERT_EQ(store.Size(), 1);
  rc->RemoveReference(id1);
  ASSERT_EQ(store.Size(), 0);

  // Tests putting an object with no references is ignored.
  RAY_CHECK_OK(store.Put(id2, buffer));
  ASSERT_EQ(store.Size(), 0);

  // Tests ref counting overrides remove after get option.
  rc->AddReference(id1);
  RAY_CHECK_OK(store.Put(id1, buffer));
  ASSERT_EQ(store.Size(), 1);
  std::vector<std::shared_ptr<RayObject>> results;
  RAY_CHECK_OK(store.Get({id1}, /*num_objects*/ 1, /*timeout_ms*/ -1,
                         /*remove_after_get*/ true, &results));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(store.Size(), 1);
}

}  // namespace ray

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
